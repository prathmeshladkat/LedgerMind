import re
import uuid
import logging
from sentence_transformers import SentenceTransformer
from qdrant_client.models import PointStruct
from agents.state import FilingState
from infra.qdrant.client import get_qdrant, COLLECTION_NAME
from infra.redis.client import publish_status

logger = logging.getLogger(__name__)

_embedding_model = None

def get_embedding_model() -> SentenceTransformer:
    """
    lazy loads the model on first use.
    model is cached in _embedding_model so it loads only once.
    loading a model takes 2-3 seconds - we dont want that per request.
    """
    global _embedding_model
    if _embedding_model is None:
        logger.info("Loading sentence-transformer model...")
        _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info("Model loaded")

    return _embedding_model


SEC_SECTION_PATTERNS = [
    r"item\s+1a\.?\s+risk factors",
    r"item\s+1\.?\s+business",
    r"item\s+2\.?\s+properties",
    r"item\s+7\.?\s+management",
    r"item\s+7a\.?\s+quantitative",
    r"item\s+8\.?\s+financial statements",
    r"item\s+9a\.?\s+controls",
]

def clean_html(raw_html: str) -> str:
    """
    removes HTML tags from raw filing HTML.
    SEC filings are HTML documents - we need plain text for embedding.
    also removes excessive whitespace to reduce token count.
    """
    # remove all HTML tags
    text = re.sub(r"<[^>]+>", " ", raw_html)
    # remove multiple spaces and newlines
    text = re.sub(r"\s+", " ", text)
    # remove special HTML entities like &nbsp; &amp;
    text = re.sub(r"&[a-zA-Z]+;", " ", text)
    return text.strip()

def split_into_sections(text: str) -> list[str]:
    """
    splits filing text into sections by SEC standard headers.
    if no headers found, falls back to fixed-size chunking.

    why sections instead of fixed chunks?
    sections keep related content together.
    "Revenue decreased 10%" and its explanation stay in same chunk.
    fixed chunks might split a sentence in half.
    """
    # try to split by SEC section headers
    pattern = "|".join(SEC_SECTION_PATTERNS)
    sections = re.split(pattern, text, flags=re.IGNORECASE)

    # filter out empty sections and very short ones (less than 100 chars)
    sections = [s.strip() for s in sections if len(s.strip()) > 100]

    if len(sections) > 1:
        logger.info(f"Split into {len(sections)} sections by headers")
        return sections

    # fallback: split into fixed-size chunks of 1000 characters
    # with 200 character overlap to avoid cutting context
    logger.info("No section headers found, using fixed-size chunking")
    chunk_size = 500
    overlap = 100
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap  # overlap with previous chunk

    return chunks

async def embed_and_store_chunks(
    chunks: list[str],
    ticker: str,
    filing_type: str
) -> list[str]:
    """
    converts text chunks to vectors and stores in Qdrant.
    returns list of Qdrant point IDs so extract agent can retrieve them.

    PATTERN: RAG preparation
    we store ticker and filing_type as metadata (payload) on each point.
    this lets us filter search results by company when querying.
    """
    model = get_embedding_model()
    qdrant = get_qdrant()

    # embed all chunks at once - faster than one by one
    # encode() returns numpy array of shape (n_chunks, 384)
    logger.info(f"Embedding {len(chunks)} chunks...")
    embeddings = model.encode(chunks, show_progress_bar=False)

    # create Qdrant points
    # each point has: id, vector, payload (metadata)
    point_ids = []
    points = []

    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        point_id = str(uuid.uuid4())
        point_ids.append(point_id)

        points.append(PointStruct(
            id=point_id,
            vector=embedding.tolist(),   # numpy → python list
            payload={
                "text": chunk,           # original text for retrieval
                "ticker": ticker,
                "filing_type": filing_type,
                "chunk_index": i,        # position in original document
            }
        ))

    # store all points in Qdrant in one batch call
    # upsert = insert or update if ID already exists
    await qdrant.upsert(
        collection_name=COLLECTION_NAME,
        points=points
    )

    logger.info(f"Stored {len(points)} chunks in Qdrant")
    return point_ids


async def chunk_document_node(state: FilingState) -> dict:
    """
    main node function for chunk agent.
    reads raw_html from state, produces chunks and chunk_ids.
    """
    thread_id = state["thread_id"]
    ticker = state["ticker"]
    filing_type = state["filing_type"]

    # if fetch agent failed, raw_html is None - skip chunking
    if not state.get("raw_html"):
        error = "chunk_document skipped: no raw_html in state"
        logger.warning(error)
        return {"errors": state["errors"] + [error]}

    await publish_status(thread_id, "chunking", {
        "message": "Splitting document into sections"
    })

    try:
        # step 1: clean HTML to plain text
        plain_text = clean_html(state["raw_html"])
        logger.info(f"Cleaned text length: {len(plain_text)} chars")

        # step 2: split into sections
        chunks = split_into_sections(plain_text)
        chunks = [c[:800] for c in chunks]

        # step 3: embed and store in Qdrant
        chunk_ids = await embed_and_store_chunks(chunks, ticker, filing_type)

        await publish_status(thread_id, "chunk_complete", {
            "message": f"Split into {len(chunks)} sections",
            "chunk_count": len(chunks)
        })

        return {
            "chunks": chunks,
            "chunk_ids": chunk_ids,
        }

    except Exception as e:
        error_msg = f"chunk_document failed: {str(e)}"
        logger.error(error_msg)
        await publish_status(thread_id, "chunk_failed", {"message": error_msg})
        return {"errors": state["errors"] + [error_msg]}