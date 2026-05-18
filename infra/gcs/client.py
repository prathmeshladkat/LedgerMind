# infra/gcs/client.py
# manages Google Cloud Storage connection
# we store raw filing HTML here before processing


from google.cloud import storage
from infra.settings import get_settings
import asyncio
import logging

logger = logging.getLogger(__name__)

_bucket = None


def init_gcs():
    """
    creates GCS client and gets reference to our bucket.
    reads credentials from GOOGLE_APPLICATION_CREDENTIALS path.
    call this at startup (skip if gcp-key.json not ready yet).
    """
    global _bucket
    settings = get_settings()

    client = storage.Client()
    _bucket = client.bucket(settings.gcs_bucket_name)
    logger.info(f"GCS initialized: bucket={settings.gcs_bucket_name}")


def get_bucket():
    """returns bucket. raises if not initialized."""
    if _bucket is None:
        raise RuntimeError("GCS not initialized. Call init_gcs() at startup.")
    return _bucket


async def upload_filing(ticker: str, filing_type: str, content: str) -> str:
    """
    uploads raw filing HTML to GCS.
    GCS client is synchronous so we run it in a thread pool
    with run_in_executor so it doesn't block our async app.
    returns the blob path so we can download it later.
    """
    bucket = get_bucket()
    blob_path = f"filings/{ticker}/{filing_type}/raw.html"
    blob = bucket.blob(blob_path)

    loop = asyncio.get_event_loop()
    await loop.run_in_executor(
        None,  # uses default thread pool
        lambda: blob.upload_from_string(content, content_type="text/html")
    )

    logger.info(f"Uploaded to GCS: {blob_path}")
    return blob_path


async def download_filing(blob_path: str) -> str:
    """
    downloads raw filing HTML from GCS.
    same pattern - sync GCS client runs in thread pool.
    """
    bucket = get_bucket()
    blob = bucket.blob(blob_path)

    loop = asyncio.get_event_loop()
    content = await loop.run_in_executor(None, blob.download_as_text)
    return content