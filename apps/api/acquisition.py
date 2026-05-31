"""
    This is Playwright, a module that allows us to open 
    a local browser, navigate to the Tic Tac Go page on 
    Google, and capture a screenshot of the game board.
"""


import os
import tempfile
import logging
from pathlib import Path


DEFAULT_GOOGLE_TIC_TAC_GO_URL = "https://www.google.com/search?q=tic+tac+go&hl=en&gl=US"
logger = logging.getLogger("tic_tac_go.daily_solve")


class BoardCaptureError(RuntimeError):
    """Raised when the Google board cannot be captured."""


def _dismiss_tutorial_overlay(page) -> bool:
    """Click the Google Tic Tac Go tutorial Skip button when it appears."""
    candidates = [
        page.get_by_text("Skip", exact=True),
        page.get_by_role("button", name="Skip"),
        page.locator("text=Skip"),
    ]

    for locator in candidates:
        try:
            if locator.count() == 0:
                continue
            first = locator.first
            first.wait_for(state="visible", timeout=2_000)
            first.click(timeout=2_000)
            logger.info("capture.tutorial_dismissed selector=%s", locator)
            page.wait_for_timeout(1_000)
            return True
        except Exception as exc:
            logger.info("capture.skip_click_candidate_failed error=%s", exc)

    logger.info("capture.tutorial_not_found")
    return False


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

    logger.info("capture.start url=%s screenshot_path=%s", url, screenshot_path)

    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            logger.info("capture.browser_launched chromium=headless")
            page = browser.new_page(
                viewport={"width": 1280, "height": 1400},
                device_scale_factor=1,
            )
            page.goto(url, wait_until="networkidle", timeout=45_000)
            logger.info("capture.page_loaded title=%r current_url=%s", page.title(), page.url)
            page.wait_for_timeout(2_000)
            dismissed = _dismiss_tutorial_overlay(page)
            logger.info("capture.tutorial_dismissed=%s", dismissed)
            page.screenshot(path=str(screenshot_path), full_page=True)
            browser.close()
    except PlaywrightTimeoutError as exc:
        raise BoardCaptureError(f"Timed out while loading Google Tic Tac Go: {url}") from exc
    except Exception as exc:
        raise BoardCaptureError(f"Could not capture Google Tic Tac Go: {exc}") from exc

    if not screenshot_path.is_file():
        raise BoardCaptureError("Playwright finished without producing a screenshot.")

    logger.info(
        "capture.done screenshot_path=%s bytes=%s",
        screenshot_path,
        screenshot_path.stat().st_size,
    )
    return screenshot_path
