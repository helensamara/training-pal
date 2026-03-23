"""
scripts/sync_sugarwod.py
Automates: SugarWOD export request → wait for Gmail → download CSV → data/workouts.csv

Usage:
    python scripts/sync_sugarwod.py

First run opens a visible browser so you can complete any 2FA prompts.
After logging in once, the session is saved in .browser_profile/ and
subsequent runs are fully automatic.
"""
import os
import time
import shutil
from pathlib import Path
from dotenv import load_dotenv
from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout

load_dotenv()

SUGARWOD_EMAIL    = os.environ['SUGARWOD_EMAIL']
SUGARWOD_PASSWORD = os.environ['SUGARWOD_PASSWORD']
GMAIL_PASSWORD    = os.environ['GMAIL_PASSWORD']

ROOT          = Path(__file__).parent.parent
DATA_DIR      = ROOT / 'data'
PROFILE_DIR   = ROOT / '.browser_profile'
DOWNLOAD_DIR  = ROOT / '.browser_profile' / 'downloads'
DEST_CSV      = DATA_DIR / 'workouts.csv'

DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Clean up stale Chromium lock files from a previous crashed run
for lock in ('SingletonLock', 'SingletonCookie', 'SingletonSocket'):
    (PROFILE_DIR / lock).unlink(missing_ok=True)


def _log(msg):
    print(f'  {msg}')


def _download_from_gmail():
    """Use IMAP to find the latest SugarWOD export email and download the CSV."""
    import imaplib
    import email as email_lib
    from email.header import decode_header

    _log('Connecting to Gmail via IMAP...')
    mail = imaplib.IMAP4_SSL('imap.gmail.com')
    mail.login(SUGARWOD_EMAIL, GMAIL_PASSWORD)
    mail.select('inbox')

    for attempt in range(8):
        _, data = mail.search(None, 'FROM', '"hello@sugarwod.com"')
        ids = data[0].split()
        if ids:
            # Get the most recent one
            _, msg_data = mail.fetch(ids[-1], '(RFC822)')
            msg = email_lib.message_from_bytes(msg_data[0][1])
            for part in msg.walk():
                fname = part.get_filename()
                if fname and fname.endswith('.csv'):
                    payload = part.get_payload(decode=True)
                    DEST_CSV.write_bytes(payload)
                    _log(f'✓ Downloaded {fname} → {DEST_CSV}')
                    mail.logout()
                    return DEST_CSV
            _log('  Email found but no CSV attachment yet.')
        else:
            _log(f'  Email not found yet, retrying in 15 s (attempt {attempt + 1}/8)...')
        time.sleep(15)

    mail.logout()
    return None


def sync():
    with sync_playwright() as p:
        context = p.chromium.launch_persistent_context(
            user_data_dir=str(PROFILE_DIR),
            headless=False,
            accept_downloads=True,
            downloads_path=str(DOWNLOAD_DIR),
            args=['--start-maximized'],
        )

        # ── Phase 1: SugarWOD export ───────────────────────────────
        _log('Opening SugarWOD...')
        page = context.new_page()
        page.goto('https://app.sugarwod.com/', wait_until='networkidle')

        # Log in if not already logged in
        if 'login' in page.url or page.query_selector('input[type="email"]'):
            _log('Logging in to SugarWOD...')
            # Use type() instead of fill() so React/Vue forms detect the input
            email_field = page.locator('input[type="email"]')
            email_field.click()
            email_field.type(SUGARWOD_EMAIL)
            pw_field = page.locator('input[type="password"]')
            pw_field.click()
            pw_field.type(SUGARWOD_PASSWORD)
            # Wait for the submit button to become enabled
            page.wait_for_selector('button[type="submit"]:not([disabled])', timeout=10_000)
            page.click('button[type="submit"]')
            page.wait_for_load_state('networkidle', timeout=30_000)
            _log('✓ Logged in')
        else:
            _log('✓ Already logged in (session restored)')

        # Dismiss cookie consent if present
        try:
            page.click('button:has-text("Accept"), button:has-text("Accept all")', timeout=4_000)
            _log('✓ Cookie consent dismissed')
            time.sleep(1)
        except PWTimeout:
            pass

        # Navigate to athlete profile where Export Workouts lives
        _log('Navigating to profile page...')
        page.goto('https://app.sugarwod.com/athletes/me', wait_until='networkidle')
        time.sleep(1)

        # Dismiss cookie consent again if it reappeared
        try:
            page.click('button:has-text("Accept"), button:has-text("Accept all")', timeout=3_000)
            time.sleep(1)
        except PWTimeout:
            pass

        # Click Export Workouts
        _log('Clicking Export Workouts...')
        page.click('text=Export Workouts', timeout=10_000)
        time.sleep(1)

        # The confirm button is "Ok!" — force-click since it may be inside a modal
        _log('Confirming export...')
        try:
            page.wait_for_selector('button:has-text("Ok!"), a:has-text("Ok!")', timeout=8_000)
            page.click('button:has-text("Ok!"), a:has-text("Ok!")', timeout=5_000)
        except PWTimeout:
            # Try force-clicking the hidden button
            page.evaluate("document.querySelector('[data-dismiss], button.btn-primary').click()")
        _log('✓ Export requested — SugarWOD will email the CSV')

        context.close()

        # ── Phase 2: Gmail via IMAP ────────────────────────────────
        _log('Waiting 20 s for SugarWOD email to arrive...')
        time.sleep(20)
        csv_path = _download_from_gmail()

        if csv_path:
            print(f'\n✅  Done! workouts.csv updated at {csv_path}')
        else:
            print('\n⚠️  Could not find the SugarWOD email after 2 minutes.')
            print('   Check your inbox manually and re-run if needed.')


if __name__ == '__main__':
    sync()
