"""
scripts/sync_powerlifting.py
Downloads powerlifting program PDFs from Facebook Messenger (Tom Kean's chat).

All files are named "Helen.pdf" — this script renames each with the date
Tom sent it: data/powerlifting/program_YYYY-MM-DD.pdf

If Facebook shows a CAPTCHA or 2FA, the browser window stays open so you
can solve it manually, then press Enter in the terminal to continue.

Run with:
    python scripts/sync_powerlifting.py
"""
import os
import re
import time
import shutil
from pathlib import Path
from dotenv import load_dotenv
from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout

load_dotenv()

FACEBOOK_EMAIL    = os.environ['FACEBOOK_EMAIL']
FACEBOOK_PASSWORD = os.environ['FACEBOOK_PASSWORD']
COACH_NAME        = 'Tom Kean'

ROOT          = Path(__file__).parent.parent
DEST_DIR      = ROOT / 'data' / 'powerlifting'
DOWNLOAD_DIR  = ROOT / '.browser_profile' / 'downloads'
PROFILE_DIR   = ROOT / '.browser_profile'

DEST_DIR.mkdir(parents=True, exist_ok=True)
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

for lock in ('SingletonLock', 'SingletonCookie', 'SingletonSocket'):
    (PROFILE_DIR / lock).unlink(missing_ok=True)


def _log(msg):
    print(f'  {msg}')


def _pause_for_human(page, reason):
    """Pause and let the user interact with the browser, then continue."""
    print(f'\n  ⚠️  {reason}')
    print('  Solve it in the browser window, then press Enter here to continue...')
    input('  > ')
    page.wait_for_load_state('domcontentloaded', timeout=30_000)
    time.sleep(2)


def _wait_for_messenger(page):
    """Wait until Messenger inbox is visible, handling CAPTCHAs along the way."""
    for _ in range(120):
        url = page.url
        if 'messenger.com' in url and 'login' not in url:
            # Check we're actually in the inbox (not a loading screen)
            if page.query_selector('[aria-label*="Chats"], [aria-label*="New message"], input[placeholder*="Search"]'):
                return True
        if page.query_selector('iframe[src*="recaptcha"], .g-recaptcha, #captcha'):
            _pause_for_human(page, 'Facebook is showing a CAPTCHA.')
        if 'checkpoint' in url or 'login' in url:
            if page.query_selector('input[name="email"]'):
                _log('Not logged in yet, filling credentials...')
                page.fill('input[name="email"]', FACEBOOK_EMAIL)
                page.fill('input[name="pass"]', FACEBOOK_PASSWORD)
                page.click('button[name="login"]')
        time.sleep(1)
    return False


def _parse_date(text):
    """Extract YYYY-MM-DD from a timestamp string."""
    if not text:
        return None
    text = str(text)
    try:
        import pandas as pd
        for pat in [
            r'\d{4}-\d{2}-\d{2}',
            r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4}',
            r'\d{1,2}/\d{1,2}/\d{4}',
            r'\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4}',
        ]:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                return pd.to_datetime(m.group(0), dayfirst=False).strftime('%Y-%m-%d')
    except Exception:
        pass
    return None


def _already_downloaded():
    """Return set of dates already downloaded."""
    dates = set()
    for f in DEST_DIR.glob('program_*.pdf'):
        m = re.search(r'(\d{4}-\d{2}-\d{2})', f.name)
        if m:
            dates.add(m.group(1))
    return dates


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

        # ── Step 1: Get to Messenger ───────────────────────────────
        _log('Opening Messenger...')
        page.goto('https://www.messenger.com/', wait_until='domcontentloaded')
        time.sleep(3)

        if not _wait_for_messenger(page):
            _pause_for_human(page, 'Could not reach Messenger inbox automatically. Please log in manually.')
            if not _wait_for_messenger(page):
                print('❌  Could not reach Messenger. Please try again.')
                context.close()
                return

        _log('✓ Messenger loaded')

        # ── Step 2: Open Tom Kean's conversation ──────────────────
        _log(f'Opening conversation with {COACH_NAME}...')
        opened = False
        # Try clicking from the conversation list
        for attempt in range(3):
            try:
                page.click(f'text="{COACH_NAME}"', timeout=4_000)
                opened = True
                break
            except PWTimeout:
                pass
            try:
                page.click(f'text={COACH_NAME}', timeout=4_000)
                opened = True
                break
            except PWTimeout:
                pass
            # Try search
            try:
                search = page.locator('input[placeholder*="Search"]').first
                search.click(timeout=4_000)
                search.fill('')
                search.type(COACH_NAME, delay=80)
                time.sleep(2)
                page.click(f'text={COACH_NAME}', timeout=6_000)
                opened = True
                break
            except PWTimeout:
                time.sleep(2)

        if not opened:
            _pause_for_human(page, f'Could not find {COACH_NAME} automatically. Please click on the conversation manually.')

        time.sleep(2)
        _log('✓ Conversation open')

        # ── Step 3: Open the Files panel ──────────────────────────
        _log('Opening Files panel (info → Shared Files)...')
        screenshot_path = str(PROFILE_DIR / 'before_files.png')
        page.screenshot(path=screenshot_path)

        files_opened = False
        # Try the info / settings button at the top right of the conversation
        for selector in [
            '[aria-label="Conversation information"]',
            '[aria-label="Info"]',
            '[aria-label*="information" i]',
            '[aria-label*="details" i]',
            'svg[aria-label*="info" i]',
        ]:
            try:
                page.click(selector, timeout=3_000)
                files_opened = True
                break
            except PWTimeout:
                continue

        if not files_opened:
            _pause_for_human(page, 'Please click the ⓘ info button (top right of chat) to open the conversation info panel.')

        time.sleep(1)

        # Click "Files" or "Shared Files" section
        for selector in ['text=Files', 'text=Shared Files', '[aria-label*="Files"]', 'text=See all files']:
            try:
                page.click(selector, timeout=4_000)
                time.sleep(1)
                break
            except PWTimeout:
                continue

        # ── Step 4: Download all Helen.pdf files ──────────────────
        _log('Scanning for PDF files...')
        time.sleep(2)
        downloaded = []
        unknown_count = 0

        # Keep scrolling through the files list and downloading
        for scroll_round in range(200):
            # Find all file entries on screen
            file_entries = page.query_selector_all(
                '[aria-label*=".pdf" i], [aria-label*="Helen.pdf" i], '
                'a[href*=".pdf"], [role="row"]:has-text(".pdf"), '
                '[class*="file"]:has-text("Helen")'
            )

            new_this_round = 0
            for entry in file_entries:
                try:
                    # Get timestamp from nearest element
                    date_str = None
                    entry.hover()
                    time.sleep(0.3)

                    # Try to get date from tooltip or nearby text
                    for attr in ['aria-label', 'title', 'data-tooltip-content']:
                        val = entry.get_attribute(attr)
                        if val:
                            date_str = _parse_date(val)
                            if date_str:
                                break

                    # Look for date text near the entry
                    if not date_str:
                        parent = page.evaluate_handle(
                            'el => el.closest("[role=row], [class*=item], [class*=file]") || el.parentElement',
                            entry
                        )
                        if parent:
                            parent_text = page.evaluate('el => el?.innerText || ""', parent)
                            date_str = _parse_date(parent_text)

                    # Check tooltip
                    if not date_str:
                        tt = page.query_selector('[role="tooltip"]')
                        if tt:
                            date_str = _parse_date(tt.inner_text())

                    if date_str and date_str in existing:
                        continue  # already have it

                    fname = f'program_{date_str}.pdf' if date_str else f'program_unknown_{unknown_count + 1:03d}.pdf'
                    dest  = DEST_DIR / fname
                    if dest.exists():
                        continue

                    # Click to download
                    _log(f'  Downloading: {fname}')
                    with page.expect_download(timeout=20_000) as dl:
                        entry.click()
                    download = dl.value
                    shutil.copy(download.path(), dest)
                    _log(f'  ✓ Saved: {fname}')
                    downloaded.append(fname)
                    if date_str:
                        existing.add(date_str)
                    else:
                        unknown_count += 1
                    new_this_round += 1
                    time.sleep(0.5)

                except Exception as e:
                    _log(f'  Skipped: {e}')
                    continue

            # Scroll down in the files panel to load more
            page.keyboard.press('End')
            time.sleep(1)
            prev_h = page.evaluate('document.body.scrollHeight')
            page.evaluate('window.scrollBy(0, 800)')
            time.sleep(1)
            new_h = page.evaluate('document.body.scrollHeight')
            if new_h == prev_h and new_this_round == 0:
                break  # nothing new

        context.close()

    if downloaded:
        print(f'\n✅  Done! Downloaded {len(downloaded)} new program(s):')
        for f in downloaded:
            print(f'    {f}')
    else:
        print('\n  No new PDFs downloaded.')
        pdfs = list(DEST_DIR.glob('program_*.pdf'))
        if pdfs:
            print(f'  Already have {len(pdfs)} programs — you\'re up to date!')


if __name__ == '__main__':
    sync()
