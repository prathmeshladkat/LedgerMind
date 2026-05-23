import httpx
import logging
from datetime import datetime
from agents.state import FilingState
from infra.redis.client import publish_status
from infra.gcs.client import upload_filing

logger = logging.getLogger(__name__)

EDGAR_BASE_URL = "https://data.sec.gov"   # keep this for submissions
SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"  

EDGAR_HEADERS = {
    "User-Agent": "CapitalSense research@capitalsense.ai",
    "Accept-Encoding": "gzip, deflate",
}

async def get_cik_for_ticker(ticker: str) -> str:
    """
    SEC EDGAR identifies companies by CIK numbe, not ticker.
    this function converts ticker -> CIK.
    e.g. AAPL -> 0000320193

    SEC provides a public JSON file that maps all tickers to CIKs.
    we fetch it once per request (could be cached in Redis later).
    """
    url = "https://www.sec.gov/files/company_tickers.json"

    async with httpx.AsyncClient(headers=EDGAR_HEADERS) as client:
        response = await client.get(url)
        response.raise_for_status()
        data = response.json()

     # the JSON is a dict where each value has ticker and cik_str
    # we search for our ticker and return the CIK
    for entry in data.values():
        if entry["ticker"].upper() == ticker.upper():
            # CIK must be padded to 10 digits with leading zeros
            return str(entry["cik_str"]).zfill(10)

    raise ValueError(f"Ticker {ticker} not found in SEC EDGAR")

async def get_latest_filing_url(cik: str, filing_type: str) -> tuple[str, str, str]:
    """
    finds the most recent filing and returns the actual document URL.
    """
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"

    async with httpx.AsyncClient() as client:
        response = await client.get(url, headers=EDGAR_HEADERS, timeout=30.0)
        response.raise_for_status()
        data = response.json()

    filings = data["filings"]["recent"]
    forms = filings["form"]
    dates = filings["filingDate"]
    accessions = filings["accessionNumber"]
    primary_docs = filings.get("primaryDocument", [])

    for i, form in enumerate(forms):
        if form == filing_type:
            accession_clean = accessions[i].replace("-", "")
            date = dates[i]
            cik_int = int(cik)

            # get primary document name (e.g. nvda-20250126.htm)
            primary_doc = primary_docs[i] if i < len(primary_docs) else None

            if primary_doc:
                # direct link to the actual filing document
                doc_url = (
                    f"https://www.sec.gov/Archives/edgar/data/"
                    f"{cik_int}/{accession_clean}/{primary_doc}"
                )
            else:
                # fallback to index page
                doc_url = (
                    f"https://www.sec.gov/Archives/edgar/data/"
                    f"{cik_int}/{accession_clean}/{accessions[i]}-index.htm"
                )

            return doc_url, date, accessions[i]

    raise ValueError(f"No {filing_type} found for CIK {cik}")


async def download_filing_html(url: str) -> str:
    """
    downloads the raw HTML of a SEC filing.
    some filings are very large (5-10MB) so we set a long timeout.
    """
    async with httpx.AsyncClient(
        headers=EDGAR_HEADERS,
        timeout=60.0,      # 60 second timeout for large filings
        follow_redirects=True
    ) as client:
        response = await client.get(url)
        response.raise_for_status()
        return response.text


async def fetch_filing_node(state: FilingState) -> dict:
    """
    main node function - LangGraph calls this automatically.
    receives current state, returns dict of fields to update.

    IMPORTANT: only return the fields this node changes.
    LangGraph merges your returned dict into the full state.
    returning the full state causes duplicate data bugs.

    flow:
    ticker → CIK → latest filing URL → download HTML → save to GCS
    """
    ticker = state["ticker"]
    filing_type = state["filing_type"]
    thread_id = state["thread_id"]

    logger.info(f"Fetching {filing_type} for {ticker}")

    # tell the UI we started - WebSocket subscribers see this
    # PATTERN: Event-Driven UI Updates via Redis pub/sub
    await publish_status(thread_id, "fetching", {
        "message": f"Downloading {filing_type} for {ticker}"
    })

    try:
        # step 1: get CIK from ticker
        cik = await get_cik_for_ticker(ticker)
        logger.info(f"CIK for {ticker}: {cik}")

        # step 2: find latest filing URL
        filing_url, filing_date, accession_number = await get_latest_filing_url(
            cik, filing_type
        )
        logger.info(f"Filing URL: {filing_url}")

        # step 3: download the HTML
        raw_html = await download_filing_html(filing_url)
        logger.info(f"Downloaded {len(raw_html)} characters")

        # step 4: save to GCS for later reference
        # PATTERN: we save raw data before processing
        # if extraction fails later, we don't re-download
        gcs_path = await upload_filing(ticker, filing_type, raw_html)

        # tell UI fetch is done
        await publish_status(thread_id, "fetch_complete", {
            "message": f"Downloaded {filing_type} for {ticker}",
            "filing_date": filing_date
        })

        # return only the fields this node is responsible for
        return {
            "raw_html": raw_html,
            "gcs_blob_path": gcs_path,
            "filing_date": filing_date,
            "accession_number": accession_number,
        }

    except Exception as e:
        # log error but don't crash the whole graph
        # add to errors list so validator can see what went wrong
        error_msg = f"fetch_filing failed: {str(e)}"
        logger.error(error_msg)

        await publish_status(thread_id, "fetch_failed", {
            "message": error_msg
        })

        # return error - LangGraph merges this into state
        # validate node will see errors list is not empty
        return {
            "errors": state["errors"] + [error_msg],
            "raw_html": None,
        }  