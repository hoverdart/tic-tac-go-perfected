"""
    This is Playwright, a module that allows us to open 
    a local browser, navigate to the Tic Tac Go page on 
    Google, and capture a screenshot of the game board.
"""


import os
import tempfile
from pathlib import Path


DEFAULT_GOOGLE_TIC_TAC_GO_URL = "https://www.google.com/search?q=tic+tac+go&hl=en&gl=US"


class BoardCaptureError(RuntimeError):
    """Raised when the Google board cannot be captured."""


def google_tic_tac_go_url() -> str:
    return os.getenv("GOOGLE_TIC_TAC_GO_URL", DEFAULT_GOOGLE_TIC_TAC_GO_URL)


def capture_google_board_screenshot(source_url: str | None = None) -> Path:
    try:
        from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise BoardCaptureError(
            "Missing dependency: install Playwright with "
            "`python3 -m pip install playwright` and run `playwright install chromium`."
        ) from exc

    url = source_url or google_tic_tac_go_url()
    output_dir = Path(tempfile.mkdtemp(prefix="tic-tac-go-"))
    screenshot_path = output_dir / "google-tic-tac-go.png"

    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            page = browser.new_page( viewport={"width": 1280, "height": 1400},device_scale_factor=1,)
            page.goto(url, wait_until="networkidle", timeout=45_000)
            page.wait_for_timeout(2_000)
            page.screenshot(path=str(screenshot_path), full_page=True)
            browser.close()
    except PlaywrightTimeoutError as exc:
        raise BoardCaptureError(f"Timed out while loading Google Tic Tac Go: {url}") from exc
    except Exception as exc:
        raise BoardCaptureError(f"Could not capture Google Tic Tac Go: {exc}") from exc

    if not screenshot_path.is_file():
        raise BoardCaptureError("Playwright finished without producing a screenshot.")

    return screenshot_path
