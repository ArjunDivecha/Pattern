"""
Script: share_export.py
Version: 1.0 (2026-10-18)

Description:
    Turns the shareable strategy page (share_page.py output) into files that open inside WhatsApp
    without any login: one long single-page PDF of the whole page, and five PNG cards (summary,
    what the network sees, growth of $1, the decile staircase, year by year) at phone width and
    2x pixel density. Renders with headless Chromium via Playwright, after the charts have drawn.
    PNG is used for the cards because WhatsApp shows images inline; the PDF carries everything.

Input files:
    <run>/share_page_<UTC>/pattern_cnn_share.html  (newest, or --page)
    with <run> = /Users/arjundivecha/Dropbox/AAA Backup/A Working/Pattern/research/r3000_headline/runs/headline_I20R20_v2_cloud_20261012T034158Z
Output files (next to the page, in whatsapp/):
    pattern_cnn_backtest.pdf, 1_summary.png, 2_what_it_sees.png, 3_growth.png, 4_staircase.png, 5_years.png
    and a copy of the folder at /Users/arjundivecha/Desktop/Pattern WhatsApp/

Usage:
    python3 research/r3000_headline/share_export.py [--page PATH]

Dependencies: playwright (Python) with its Chromium.
"""
from pathlib import Path
import argparse
import shutil

from playwright.sync_api import sync_playwright

RUN = Path(__file__).resolve().parent / 'runs/headline_I20R20_v2_cloud_20261012T034158Z'
DESKTOP = Path('/Users/arjundivecha/Desktop/Pattern WhatsApp')
CARDS = [('1_summary.png', 'header'), ('2_what_it_sees.png', 'section:nth-of-type(2)'),
         ('3_growth.png', 'section:nth-of-type(3)'), ('4_staircase.png', 'section:nth-of-type(4)'),
         ('5_years.png', 'section:nth-of-type(5)')]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--page', default=None)
    a = ap.parse_args()
    page_path = Path(a.page) if a.page else sorted(RUN.glob('share_page_*/pattern_cnn_share.html'))[-1]
    out = page_path.parent / 'whatsapp'
    out.mkdir(exist_ok=True)
    html = page_path.read_text()
    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_page(viewport={'width': 560, 'height': 1000}, device_scale_factor=2)
        pg.set_content(html, wait_until='networkidle')
        pg.wait_for_function('window.Chart && document.querySelectorAll("canvas").length === 4')
        pg.wait_for_timeout(1500)
        pg.add_style_tag(content='.wrap > header, .wrap > section { padding: 20px 22px; background: #fbfcfb; }')
        pg.wait_for_timeout(800)
        # Cards: the sections are inside .wrap; header is the first child.
        for name, sel in CARDS:
            el = pg.locator('.wrap > ' + sel).first
            el.screenshot(path=str(out / name), animations='disabled')
        # One long PDF page at the same width, charts as drawn on screen.
        pg.emulate_media(media='screen')
        h = pg.evaluate('document.documentElement.scrollHeight')
        pg.pdf(path=str(out / 'pattern_cnn_backtest.pdf'), width='560px', height=f'{h + 40}px',
               print_background=True, margin={'top': '0', 'bottom': '0', 'left': '0', 'right': '0'})
        b.close()
    if DESKTOP.exists():
        shutil.rmtree(DESKTOP)
    shutil.copytree(out, DESKTOP)
    for f in sorted(out.iterdir()):
        print(f, f.stat().st_size)


if __name__ == '__main__':
    main()
