"""Local-only evidence reconciliation and readable report for the 30-case audit."""
from pathlib import Path
import sys,json,html
import pandas as pd
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,'/Users/arjundivecha/Dropbox/AAA Backup/A Working/Global Data Mart')
from gdm import dataset
c=pd.read_csv(ROOT/'cases.csv');w=pd.read_csv(ROOT/'surrounding_prices.csv');e=pd.read_csv(ROOT/'nearby_events.csv')
cache=dataset('equities.research_vintage_cache')/'prices'/'securities'
checks=[]
for _,r in c.iterrows():
    f=cache/(r.security.replace(' ','_').replace('/','-')+'.parquet')
    check={'case_id':r.case_id,'cache_path':str(f),'exists':f.exists(),'matches':False}
    if f.exists():
        d=pd.read_parquet(f);d=d[pd.to_datetime(d.date).eq(pd.Timestamp(r.date))]
        target=w[w.case_id.eq(r.case_id)&w.date.eq(r.date)]
        fields=['open','high','low','close','volume','total_return_index_gross_dividends']
        check['cache_rows_on_date']=len(d)
        if len(d)==1:
            check['matches']=bool(np.allclose(d[fields].iloc[0].to_numpy(float),target[fields].iloc[0].to_numpy(float),equal_nan=True,rtol=1e-10,atol=1e-10))
    checks.append(check)
pd.DataFrame(checks).to_csv(ROOT/'cache_reconciliation.csv',index=False)
# Findings specific to these observations, not claims about the whole data set.
for i,r in c.iterrows():
    if r.category=='Dividend control':
        c.loc[i,'classification']='Pass: stored cash dividend reconciles to total return'
        c.loc[i,'finding']=f"TRI return minus (close + cash dividend)/previous close - 1 = {r.tri_minus_cash_return_bp:.6f} basis points. This validates arithmetic for this event only, using the same provider's stored records."
    if r.category=='Split control':
        event=e[e.case_id.eq(r.case_id)&e.event_table.eq('splits')&e['Ex-Date'].eq(r.date)]
        if len(event)==1:
            ratio=float(event.iloc[0]['Dividend Amount'])
            c.loc[i,'classification']='Pass: stored split event consistent with adjusted continuity'
            c.loc[i,'finding']=f"Stored event records {ratio:g}-for-1 split on this date. Previous close {r.previous_close:g}, event-day close {r.close:g}; ordinary price return {r.price_return:.4%}. History is consistent with split-adjustment, not raw nominal prices. Do not apply the split a second time. Volume adjustment and full-history cash/special-distribution settings remain unverified."
c.to_csv(ROOT/'reviewed_cases.csv',index=False)
intro='''# Global Data Mart: 30-case local price audit

Date: 2026-09-17. Scope: existing local files only. No Bloomberg API calls, no source changes, no eligibility filters applied. This report does not use external web evidence. A few public-source searches were made before the user's local-only clarification; they are excluded from the findings.

## Findings

- Ten corporate-action controls pass limited local consistency checks: five splits have matching stored events and adjusted price continuity; five dividends reconcile to total-return arithmetic within 0.002 basis points.
- Five selected OHLC cases contain internal contradictions. Example: ASTH on 2013-11-07 has open/high/low 100.10 but close 5.35. KITDQ on 2005-08-24 has high/low 73.5011/61.2509 but close 6.3438. The correct replacement prices are NOT established.
- Five selected nonpositive observations cannot be used as ordinary positive-price inputs. A zero could represent a terminal event or a placeholder; the local evidence does not decide which. Do not delete a terminal loss from a portfolio ledger.
- All five selected extreme jumps start from a stored close of 0.000001. Their arithmetic is reproducible; actual trading, executable size, and source correctness are unverified.
- Five selected continuity breaks span roughly 25-30 years. Latest security-master identifiers do not establish historical continuity. The numerical returns across them are diagnostic artifacts, not valid holding-period returns.

## Adjustment provenance

The existing producer calls `bbg.hist_batch` for six price fields without explicit adjustment parameters (GDM `sources/equities/ingest/build_research_vintage.py`, `pull_price_payload`). The existing wrapper (`OpusBloomberg/bbg.py`, `hist_batch`) sets dates and periodicity but does not explicitly set `adjustmentFollowDPDF`, `adjustmentSplit`, `adjustmentNormal`, or `adjustmentAbnormal`. The effective historical settings are therefore not proved by the code inspected. Control events empirically support split-adjusted closes and dividend-inclusive TRI for those cases only. Volume conventions remain unresolved. Current code is not proof of the exact code version/settings used at every original retrieval.

The producer also drops rows with missing close before assembly. Consequently, zero missing closes in the final panel does not prove complete observation coverage.

## Proposed rules for your review — not applied

1. Give every source row a quality status and reason; retain raw evidence.
2. Quarantine internally inconsistent OHLC and nonpositive price inputs from chart construction pending resolution; flag any feature/label window they affect. Do not silently omit portfolio losses or exits.
3. Prevent returns and chart windows from spanning unresolved security-identity breaks. A 180-day gap is a screening threshold, not proof of a new issuer.
4. Preserve extreme positive-price observations, flag microprice/stale/zero-volume cases, and decide tradability separately. Do not winsorize returns to improve results.
5. Treat split-adjusted chart prices and total-return outcomes as distinct series; avoid applying dividends or splits twice. Obtain a documented nominal-price basis before interpreting a historical dollar-price filter.
6. The local event table named dividends also contains Stock Split rows that overlap splits.parquet. Filter by event type and deduplicate; never add a split ratio as a cash dividend.

## Evidence boundaries and sampling

This is a purposeful exception sample, not an error-rate estimate. Five distinct securities per anomaly group: largest absolute TRI returns with positive endpoints and gaps at most seven days; latest nonpositive observations; largest relative OHLC violations among closes at least 5 (for case selection only); longest gaps above 180 days. Tied dates follow input/sort order. Controls are five fixed known split dates and the last 2024 regular-cash ex-date for five fixed large-cap names. The sample uses all available history, including pre-1996 observations; it does not define the future study period. There are no separately sampled stale-price runs: zero-volume checks and surrounding observations are only preliminary stale-price evidence.

Each case includes up to three observations before and after the event. Windows are truncated at series boundaries; observation spacing is not necessarily daily. No matching event in a seven-day window is not proof that no corporate action occurred. Static security metadata is explicitly not point-in-time. Cached prices are same-source corroboration, not independent market validation.

## Files

- `reviewed_cases.csv`: final 30 classifications and calculations.
- `surrounding_prices.csv`: observed windows, including diagnostic returns.
- `nearby_events.csv`: stored events within seven calendar days.
- `security_identity.csv`: latest stored identifiers and names.
- `cache_reconciliation.csv`: comparison against per-security retrieval cache.
- `audit_metadata.json`: input file metadata and validation boundaries.
- `audit.py`, `render_report.py`: reproducible local-only audit.

## Cases
'''
parts=[intro]
cards=[]
cols=['date','open','high','low','close','volume','total_return_index_gross_dividends','gap_days']
for _,r in c.iterrows():
    rows=w[w.case_id.eq(r.case_id)][cols]
    ev=e[e.case_id.eq(r.case_id)]
    identity=f"{r['name']}; FIGI {r.id_bb_global}; ID {r.id_bb_unique}. Static metadata, not historical identity proof."
    text=f"### {r.case_id} — {r.category}: {r.security}, {r.date}\n\n**{r.classification}**\n\n{r.finding}\n\n{identity}\n\n```text\n{rows.to_string(index=False)}\n```\n\n"
    if not ev.empty:text+='Stored nearby events:\n\n```text\n'+ev[['event_table','Ex-Date','Dividend Amount','Dividend Type']].to_string(index=False)+'\n```\n\n'
    parts.append(text)
    cards.append('<section id="'+r.case_id+'"><h2>'+html.escape(f'{r.case_id} · {r.category} · {r.security} · {r.date}')+'</h2><p><strong>'+html.escape(r.classification)+'</strong></p><p>'+html.escape(r.finding)+'</p><p>'+html.escape(identity)+'</p>'+rows.to_html(index=False,na_rep='—',float_format=lambda v:f'{v:.8g}')+(ev[['event_table','Ex-Date','Dividend Amount','Dividend Type']].to_html(index=False) if not ev.empty else '<p>No matching stored event within seven days; absence is not conclusive.</p>')+'</section>')
(ROOT/'report.md').write_text('\n'.join(parts))
summary=c[['case_id','category','security','date','classification']].to_html(index=False)
(ROOT/'report.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><title>GDM 30-case price audit</title><style>body{font:16px system-ui;color:#172a35;background:#fafafa;max-width:1300px;margin:40px auto;padding:0 24px}table{border-collapse:collapse;width:100%;font-size:13px;background:white;margin:18px 0}td,th{padding:9px;border:1px solid #d9e0e3;text-align:right}th{background:#eaf0f2}section{margin:36px 0;padding:22px;background:white;border:1px solid #d9e0e3}h1,h2{line-height:1.3}pre{white-space:pre-wrap;font:15px system-ui;line-height:1.6}p{line-height:1.6}</style><h1>Global Data Mart · 30-case price audit</h1><pre>'+html.escape(intro.split('## Cases')[0].replace('#',''))+'</pre>'+summary+''.join(cards)+'</html>')
assert len(c)==30 and c.case_id.nunique()==30
assert len(checks)==30 and all(x['matches'] for x in checks),'Cache discrepancy: inspect cache_reconciliation.csv'
assert c[c.category.eq('Dividend control')].tri_minus_cash_return_bp.abs().lt(.002).all()
assert len(cards)==30
print('Validated: 30 cases, 30 cache matches, 5 dividend reconciliations within 0.002 bp, 30 HTML case sections.')
print(c.classification.value_counts().to_string())
