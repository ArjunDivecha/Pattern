"""
Script: mcap.py
Version: 1.0 (2026-10-14)

Description:
    Point-in-time market capitalisation for the R3000 headline work, shared by report.py,
    neutralize.py and cumulative_vs_r3000.py.

    Market cap = shares outstanding (BS_SH_OUT, latest quarterly filing available on or before
    the date) x the split-adjusted close. The share counts in this fundamentals vintage are
    restated to today's share basis (Apple's 1990 filing shows 14 billion shares), so they must
    be paired with the split-adjusted price, not the traded price; pairing them with the traded
    price inflated any company that later split (Apple January 2000: $2.0 trillion instead of
    ~$18 billion). That was the bug in report.py v1.0.

    The share series also holds isolated unit errors (~1,600 of ~544,000 filings jump 20x or
    more for one quarter and then return, e.g. Rayonier October 1999: 110 billion shares between
    filings of 110 million). A filing is discarded when it differs by more than 5x from both its
    previous and its next filing; a genuine, lasting change agrees with the next filing and is
    kept. The latest earlier valid filing is used instead.

Input files:
    /Users/arjundivecha/Dropbox/AAA Backup/A Working/Global Data Mart/sources/equities/data/vintages/2026-06-30/fundamentals.parquet (BS_SH_OUT)
    <dataset root>/nominal_prices.parquet (close_split_adj), dataset root from CURRENT.json
Output files: none (library module).

Usage:
    from mcap import market_caps
    caps = market_caps(root, frame_with_security_and_date)   # Series aligned to the frame's index
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
FUND = ROOT.parent / 'Global Data Mart/sources/equities/data/vintages/2026-06-30/fundamentals.parquet'
MAX_RATIO = 5.0


def clean_shares():
    sh = pd.read_parquet(FUND, filters=[('field', '==', 'BS_SH_OUT'), ('period_type', '==', 'quarterly')],
                         columns=['security', 'availability_date', 'value'])
    sh = sh.dropna()
    sh = sh[sh.value > 0].sort_values(['security', 'availability_date'])
    sh = sh.drop_duplicates(['security', 'availability_date'], keep='last').reset_index(drop=True)
    sh['security'] = sh.security.astype(str)
    g = sh.groupby('security').value
    prev, nxt = g.shift(1), g.shift(-1)

    def ok(other):
        r = sh.value / other
        return other.isna() | ((r <= MAX_RATIO) & (r >= 1 / MAX_RATIO))
    has_neighbour = prev.notna() | nxt.notna()
    agrees = (ok(prev) & prev.notna()) | (ok(nxt) & nxt.notna())
    keep = ~has_neighbour | agrees
    return sh[keep], int((~keep).sum())


def market_caps(root, frame):
    """Market cap for each (security, date) row of frame; NaN where no valid filing or price."""
    sh, _ = clean_shares()
    f = frame[['security', 'date']].copy()
    f['security'] = f.security.astype(str)
    f['_row'] = np.arange(len(f))
    secs = sorted(f.security.unique())
    px = pd.read_parquet(Path(root) / 'nominal_prices.parquet', columns=['security', 'date', 'close_split_adj'],
                         filters=[('security', 'in', secs)])
    px['security'] = px.security.astype(str)
    f = f.merge(px, on=['security', 'date'], how='left', validate='many_to_one')
    f = pd.merge_asof(f.sort_values('date'), sh.rename(columns={'availability_date': 'date', 'value': 'shares'}).sort_values('date'),
                      on='date', by='security', direction='backward')
    f = f.sort_values('_row')
    cap = f.shares.to_numpy() * f.close_split_adj.to_numpy()
    return pd.Series(cap, index=frame.index)
