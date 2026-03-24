"""
scripts/sync_powerlifting.py
Downloads powerlifting program PDFs from Facebook Messenger (Tom Kean's chat).

Strategy: scroll through the conversation messages (not the Files panel),
find every PDF attachment, read the date from the message timestamp, and
save as data/powerlifting/program_YYYY-MM-DD.pdf.

The browser stays open — log in if prompted, then press Enter to continue.

Run with:
    python scripts/sync_powerlifting.py
"""
import re
import time
import shutil
from pathlib import Path
from datetime import datetime
import pandas as pd
from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout

ROOT         = Path(__file__).parent.parent
DEST_DIR     = ROOT / 'data' / 'powerlifting'
DOWNLOAD_DIR = ROOT / '.browser_profile' / 'downloads'
PROFILE_DIR  = ROOT / '.browser_profile'

TOM_URL = 'https://www.messenger.com/e2ee/t/25785415904439619'

DEST_DIR.mkdir(parents=True, exist_ok=True)
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

for lock in ('SingletonLock', 'SingletonCookie', 'SingletonSocket'):
    (PROFILE_DIR / lock).unlink(missing_ok=True)


def _log(msg):
    print(f'  {msg}')


def _pause(msg):
    print(f'\n  ⚠️  {msg}')
    input('  Press Enter when ready...\n')


def _parse_date(text):
    """Try to extract a YYYY-MM-DD date from any text string."""
    if not text:
        return None
    text = str(text).strip()
    try:
        for pat in [
            r'\d{4}-\d{2}-\d{2}',
            r'(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}',
            r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\.?\s+\d{1,2},?\s+\d{4}',
            r'\d{1,2}/\d{1,2}/\d{4}',
            r'\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}',
        ]:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                return pd.to_datetime(m.group(0), dayfirst=False).strftime('%Y-%m-%d')
    except Exception:
        pass
    return None


def _pdf_metadata_date(pdf_path):
    """
    Extract creation date from PDF metadata.
    Tom exports his Excel file to PDF the same day he sends it, so CreationDate
    is a reliable proxy for the send date — even when Messenger hides timestamps.
    Format: D:YYYYMMDDHHmmSS[+/-HH'mm']
    """
    try:
        import pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            meta = pdf.metadata or {}
            raw = meta.get('/CreationDate', '') or meta.get('CreationDate', '') or ''
            m = re.match(r"D:(\d{4})(\d{2})(\d{2})", raw)
            if m:
                return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    except Exception:
        pass
    return None


def _already_downloaded():
    dates = set()
    for f in DEST_DIR.glob('program_*.pdf'):
        m = re.search(r'(\d{4}-\d{2}-\d{2})', f.name)
        if m:
            dates.add(m.group(1))
    return dates


def _get_message_date(page, msg_element):
    """
    Try to get the send date from a message element.
    Messenger shows dates in aria-labels or on hover tooltips.
    """
    date_str = None

    # Try the message's own aria-label (often contains timestamp)
    for attr in ['aria-label', 'data-tooltip-content', 'title']:
        try:
            val = msg_element.get_attribute(attr)
            if val:
                date_str = _parse_date(val)
                if date_str:
                    return date_str
        except Exception:
            pass

    # Hover to trigger tooltip
    try:
        msg_element.hover()
        time.sleep(0.5)
        tooltip = page.query_selector('[role="tooltip"], [data-testid="tooltip"]')
        if tooltip:
            date_str = _parse_date(tooltip.inner_text())
            if date_str:
                return date_str
    except Exception:
        pass

    # Look for a nearby timestamp span within the same message row
    try:
        parent = page.evaluate_handle(
            'el => el.closest("[role=row], [class*=message], [data-testid*=message]") || el.parentElement.parentElement',
            msg_element
        )
        if parent:
            text = page.evaluate('el => el?.innerText || ""', parent)
            date_str = _parse_date(text)
            if date_str:
                return date_str
    except Exception:
        pass

    return None


def _find_current_date_from_separator(page):
    """
    Messenger shows date separators like 'Monday, March 10' or 'February 4' in the chat.
    This gives us the date context for nearby messages.
    """
    try:
        separators = page.query_selector_all(
            '[role="separator"], [aria-label*="conversation"], '
            'div[class*="date"], span[class*="date"]'
        )
        for sep in reversed(separators):  # most recent first
            try:
                text = sep.inner_text().strip()
                if text:
                    d = _parse_date(text)
                    if d:
                        return d
            except Exception:
                continue
    except Exception:
        pass
    return None


def sync():
    existing = _already_downloaded()
    _log(f'Already have {len(existing)} programs downloaded')

    with sync_playwright() as p:
        context = p.chromium.launch_persistent_context(
            user_data_dir=str(PROFILE_DIR),
            headless=False,
            accept_downloads=True,
            downloads_path=str(DOWNLOAD_DIR),
            args=['--start-maximized'],
        )
        page = context.new_page()

        # ── Step 1: Open conversation ──────────────────────────────
        _log('Opening conversation...')
        page.goto(TOM_URL, wait_until='domcontentloaded')
        time.sleep(4)

        if 'login' in page.url or 'facebook.com' in page.url:
            _pause('Please log in to Facebook in the browser, then press Enter.')
            time.sleep(3)

        _log(f'URL: {page.url}')

        # Wait for chat to load
        for _ in range(20):
            if 'messenger.com' in page.url and 'login' not in page.url:
                break
            time.sleep(1)

        _pause(
            'The conversation should be open in the browser.\n'
            '  Make sure you can see the chat messages, then press Enter.\n'
            '  The script will scroll up automatically from here.'
        )

        # ── Step 2: Auto-scroll to top to load all messages ────────
        _log('Scrolling up to load all messages...')
        chat_box = None
        for selector in [
            '[role="main"]',
            '[aria-label*="Messages"]',
            '[class*="scrollable"]',
            'div[style*="overflow"]',
        ]:
            try:
                el = page.query_selector(selector)
                if el:
                    chat_box = selector
                    break
            except Exception:
                pass

        # Scroll up repeatedly until no more messages load
        prev_height = None
        for scroll_round in range(80):
            try:
                if chat_box:
                    page.evaluate(f'document.querySelector("{chat_box}").scrollTop = 0')
                else:
                    page.keyboard.press('Control+Home')
                time.sleep(1.5)

                # Check if new content loaded by measuring scroll height
                height = page.evaluate(
                    f'document.querySelector("{chat_box}")?.scrollHeight || document.body.scrollHeight'
                    if chat_box else 'document.body.scrollHeight'
                )
                if height == prev_height:
                    _log(f'  Reached top after {scroll_round + 1} scrolls')
                    break
                prev_height = height
                if scroll_round % 10 == 0:
                    _log(f'  Scrolling... ({scroll_round + 1} rounds)')
            except Exception as e:
                _log(f'  Scroll error (continuing): {e}')
                break

        # ── Step 3: Scan all messages for PDF attachments ──────────
        _log('Scanning messages for PDF attachments...')
        time.sleep(2)

        downloaded = []
        unknown_count = 0
        processed_ids = set()

        # Scroll down through the whole conversation, scanning as we go
        for scan_pass in range(100):
            try:
                pdf_elements = page.query_selector_all(
                    '[aria-label*=".pdf" i], '
                    '[aria-label*="Helen.pdf" i], '
                    'a[href*=".pdf"], '
                    'div[role="button"]:has-text("Helen.pdf"), '
                    'span:has-text("Helen.pdf"), '
                    '[data-testid*="file"]:has-text(".pdf")'
                )
            except Exception as e:
                _log(f'  Page closed or error during scan: {e}')
                break

            new_this_pass = 0
            for el in pdf_elements:
                try:
                    box = el.bounding_box()
                    el_id = f'{box["x"]:.0f},{box["y"]:.0f}' if box else None
                    if el_id and el_id in processed_ids:
                        continue

                    date_str = _get_message_date(page, el)
                    # Also check the aria-label of the element itself
                    if not date_str:
                        try:
                            label = el.get_attribute('aria-label') or ''
                            date_str = _parse_date(label)
                        except Exception:
                            pass

                    if date_str and date_str in existing:
                        if el_id:
                            processed_ids.add(el_id)
                        continue

                    fname = f'program_{date_str}.pdf' if date_str else f'program_unknown_{unknown_count + 1:03d}.pdf'
                    dest  = DEST_DIR / fname
                    if dest.exists():
                        if el_id:
                            processed_ids.add(el_id)
                        continue

                    _log(f'  Downloading: {fname}')
                    with page.expect_download(timeout=25_000) as dl:
                        el.click()
                    download = dl.value
                    tmp_path = download.path()
                    # If date unknown, try to extract it from PDF metadata
                    if not date_str:
                        date_str = _pdf_metadata_date(tmp_path)
                        if date_str:
                            fname = f'program_{date_str}.pdf'
                            dest  = DEST_DIR / fname
                    shutil.copy(tmp_path, dest)
                    _log(f'  ✓ Saved: {fname}')
                    downloaded.append(fname)

                    if date_str:
                        existing.add(date_str)
                    else:
                        unknown_count += 1

                    if el_id:
                        processed_ids.add(el_id)
                    new_this_pass += 1
                    time.sleep(1)

                except Exception as e:
                    _log(f'  Skipped: {e}')
                    continue

            # Scroll down a bit to reveal more messages
            try:
                if chat_box:
                    page.evaluate(
                        f'let el = document.querySelector("{chat_box}"); '
                        f'if(el) el.scrollTop += 600;'
                    )
                else:
                    page.evaluate('window.scrollBy(0, 600)')
                time.sleep(1.5)
            except Exception:
                break

            # Stop when we've scrolled to the bottom and found nothing new
            try:
                at_bottom = page.evaluate(
                    f'let el = document.querySelector("{chat_box}"); '
                    f'el ? el.scrollTop + el.clientHeight >= el.scrollHeight - 10 : true'
                    if chat_box else 'window.innerHeight + window.scrollY >= document.body.scrollHeight - 10'
                )
                if at_bottom and new_this_pass == 0:
                    _log('  Reached bottom of conversation.')
                    break
            except Exception:
                break

        context.close()

    # ── Rename unknowns if any ─────────────────────────────────────
    unknowns = sorted(DEST_DIR.glob('program_unknown_*.pdf'))
    if unknowns:
        print(f'\n  ⚠️  {len(unknowns)} PDFs could not have their date extracted automatically.')
        print('  They are saved as program_unknown_XXX.pdf')
        print('  You can rename them manually by checking the chat for when Tom sent each one.')

    if downloaded:
        print(f'\n✅  Done! Downloaded {len(downloaded)} new program(s):')
        for f in downloaded:
            print(f'    {f}')
    else:
        pdfs = list(DEST_DIR.glob('program_*.pdf'))
        print(f'\n  No new PDFs downloaded. Currently have {len(pdfs)} in data/powerlifting/')


if __name__ == '__main__':
    sync()
