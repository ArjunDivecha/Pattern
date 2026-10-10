"""Review latest successful local adjustment audit; no price corrections."""
from pathlib import Path
import sys,json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,'/Users/arjundivecha/Dropbox/AAA Backup/A Working/Global Data Mart')
from gdm import dataset

def predicted_scale_after(before, cash, adjusted_previous_close):
    denominator=1/before-cash/adjusted_previous_close
    return 1/denominator if denominator>0 else np.nan

def main():
    out=sorted(f.parent for f in ROOT.glob('adjustment_audit_*/summary.json'))[-1]
    summary=json.loads((out/'summary.json').read_text())
    c=pd.read_csv(out/'cash_events.csv',parse_dates=['date'],low_memory=False);v=pd.read_csv(out/'volume_events.csv');s=pd.read_csv(out/'split_events.csv')
    c['implied_cash']=c.prev_close*c.tri_ratio-c.close
    c['implied_scale']=c.implied_cash/c.cash
    ev=pd.read_parquet(dataset('equities.events')/'dividends.parquet');ev['date']=pd.to_datetime(ev['Ex-Date'],errors='coerce')
    evg={key:frame for key,frame in ev.groupby('security')}
    corroboration=[]
    for sec,rows in c.groupby('security'):
        regular=rows[rows.types.eq('Regular Cash')&rows.testable&rows.implied_scale.gt(0)].sort_values('date')
        specials=rows[rows.types.eq('Special Cash')&rows.testable]
        for _,r in specials.iterrows():
            before=regular[regular.date.lt(r.date)].tail(1);after=regular[regular.date.gt(r.date)].head(1)
            if before.empty or after.empty:continue
            b=before.iloc[0];a=after.iloc[0]
            if (r.date-b.date).days>200 or (a.date-r.date).days>200:continue
            interval=evg[sec];interval=interval[interval.date.gt(b.date)&interval.date.lt(a.date)]
            # Single intervening special cash, no recorded splits/spinoffs/etc.
            if len(interval)!=1 or interval.iloc[0]['Dividend Type']!='Special Cash':continue
            predicted=predicted_scale_after(b.implied_scale,r.cash,r.prev_close)
            if not np.isfinite(predicted):continue
            rel_error=a.implied_scale/predicted-1
            special_yield=r.cash/r.prev_close
            corroboration.append({'security':sec,'special_date':r.date,'special_cash':r.cash,'adjusted_prev_close':r.prev_close,'before_date':b.date,'before_scale':b.implied_scale,'after_date':a.date,'after_scale':a.implied_scale,'predicted_after_scale':predicted,'relative_error':rel_error,'scale_match_50bp':abs(rel_error)<=.005,'material_event':special_yield>=.01,'unchanged_scale_error':a.implied_scale/b.implied_scale-1})
    cr=pd.DataFrame(corroboration);cr.to_csv(out/'special_adjustment_corroboration.csv',index=False)
    c[c.security.isin(['AFG US Equity','EOG US Equity','AAPL US Equity','MSFT US Equity'])&c.date.ge('2023-01-01')].to_csv(out/'illustrative_cash_scale_histories.csv',index=False)
    material=cr[cr.material_event];stats={'isolated_special_events':len(cr),'material_special_events':len(material),'material_scale_matches_50bp':int(material.scale_match_50bp.sum()),'median_abs_relative_scale_error':float(material.relative_error.abs().median())}
    vv=v[v.testable]
    summary['special_adjustment_corroboration']=stats
    (out/'review_summary.json').write_text(json.dumps(summary,indent=2))
    nsplit=int(s.testable.sum());ncash=int(c.testable.sum());cashmatch=int(c.cash_match.sum())
    report=f'''# Stored price-adjustment conventions: local evidence

## Decision

The tested price histories are predominantly split-adjusted. They are not universally raw nominal prices, and some are consistent with additional backward adjustment for special distributions. Regular cash dividends generally remain outside the Close series and are included in the total-return index. Do not add dividends to TRI returns or apply stock splits to the existing adjusted prices a second time. No raw prices, quality-layer files, or strategy rules were changed by this audit.

This is an empirical convention assessment, not proof of every vendor setting or every row. Source requests did not record explicit adjustment settings. Current wrapper code is not proof of the exact settings at historical retrieval time. No Bloomberg/API/web calls were used in this task.

## Split evidence

- {nsplit:,} simple split events were testable: exact ex-date present, positive and internally consistent endpoints, at most seven calendar days between observations, material split factor, no other stored action on the ex-date.
- {summary['splits_by_class'].get('split_adjusted_relative_to_TRI',0):,} have TRI/price gross-return ratio within 0.5% in log terms of one, supporting split-adjusted closes relative to TRI. {summary['splits_by_class'].get('unadjusted_relative_to_TRI',0):,} support an unadjusted split jump; {summary['splits_by_class'].get('unresolved',0):,} are unresolved.
- Test is `(TRI_t/TRI_previous)/(Close_t/Close_previous)`: near one for adjusted prices, near the new/old share ratio for unadjusted prices. It assumes TRI correctly handles the action; it is same-provider corroboration.
- {summary['split_conflicting_rows']} conflicting stored split rows were excluded. Other events were not testable; exclusions are present in split_events.csv.

## Cash dividends

- {ncash:,} cash-event dates across {summary['cash_testable_securities']:,} securities were testable.
- {cashmatch:,} ({100*cashmatch/ncash:.2f}%) match `(Close_t + cash)/Close_previous - 1` to within one basis point, using either the quoted dividend or an explicitly tested later-split-normalized alternative.
- {summary['cash_by_class'].get('unresolved',0):,} remain unresolved under those simple hypotheses. This is not a source-error count. Other classes are in summary.json; tiny events can be indistinguishable at this tolerance.
- Regular-only events: {summary['cash_by_type']['Regular Cash']['matches']:,}/{summary['cash_by_type']['Regular Cash']['n']:,} match. Special-only events: {summary['cash_by_type']['Special Cash']['matches']:,}/{summary['cash_by_type']['Special Cash']['n']:,} match the simple add-cash identity. The difference motivates the special-adjustment check below.
- {summary['ambiguous_cash_event_dates']} dates with multiple same-type cash rows are excluded as ambiguous. Equal amounts can represent distinct distributions; they are not deduplicated by amount. The events table also contains splits, stock distributions and other noncash actions: its filename does not make every row cash.

## Additional historical adjustment: direct local examples

For regular dividends define `implied_cash = previous_Close * (TRI_t/TRI_previous) - Close_t`, and `scale = implied_cash / stored_cash`. A scale below one is compatible with prices on a smaller, backward-adjusted basis, but does not on its own identify the adjustment cause.

AFG shows approximately 0.9133 in January 2024, 0.9322 across April–October 2024, 0.9591 in January 2025, 0.9745 across April–October 2025, and 1.0000 by April 2026. The changes align with stored special dividends. AAPL and MSFT recent regular-dividend scales are approximately one.

To corroborate the explanation without fitting arbitrary prices, take isolated special dividends between two regular-dividend observations. If prior prices were backward-adjusted for a special payout D, then `predicted_scale_after = 1 / (1/scale_before - D/adjusted_previous_Close)`. Compare this prediction against the next regular dividend's implied scale. Require one intervening recorded event and neighboring regular events within 200 days.

- {stats['isolated_special_events']:,} isolated events permit this check; {stats['material_special_events']:,} have D/adjusted_previous_Close at least 1%.
- {stats['material_scale_matches_50bp']:,} material events match the predicted scale within 0.5%; median absolute relative discrepancy is {100*stats['median_abs_relative_scale_error']:.4f}%.

This supports additional backward adjustments for at least a subset of histories. It does not license universal de-adjustment: event coverage, cash currencies, static security identity, conflicting event records and special-action methods remain limitations. The scales are retrospective diagnostics, not point-in-time features or a proposed reconstructed price series.

## Volume

Ten positive-volume observations before and after each large split (excluding event day) provide {summary['volume_testable_events']:,} comparisons. Median post/pre volume is {summary['volume_forward']['median_ratio']:.3f} for {summary['volume_forward']['n']:,} forward splits, and {summary['volume_reverse']['median_ratio']:.3f} for {summary['volume_reverse']['n']:,} reverse splits. {summary['volume_closer_to_adjusted']:,}/{summary['volume_testable_events']:,} lie closer in log space to unchanged volume than to mechanical multiplication by the share ratio.

This is consistent with adjustment for many series, but is not proof: issuance, trading activity, illiquidity and small volumes confound it, especially around reverse splits. Do not label Close×Volume as verified historical dollars traded yet. No nominal volume reconstruction is justified by this test alone.

## Recommended study contract for your review

1. Treat OHLC as vendor-adjusted chart inputs with unresolved special-action exceptions; retain the quality flags already built.
2. Use TRI ratios once for close-to-close total-return outcomes, subject to endpoint, continuity and corporate-action checks. Do not add dividend cash again.
3. Do not interpret historical adjusted prices as executable nominal prices or use a historical $5 floor until a nominal-price convention is established.
4. Do not automatically divide cash dividends by all later split ratios: most matching events already reconcile using the quoted amounts. Normalization is a competing diagnostic hypothesis, not a blanket fix.
5. Keep volume as a vendor-provided chart feature with an unresolved units convention; do not claim validated dollar liquidity/capacity.
6. Next-open execution returns require an explicit, consistent open-price/total-return calculation and additional event checks. Close-to-close TRI validation does not automatically validate next-open P&L.

## Outputs and validation

split_events.csv, cash_events.csv, volume_events.csv, split_conflicts.csv, cash_exceptions_top100.csv, cash_by_year.csv, special_adjustment_corroboration.csv, illustrative_cash_scale_histories.csv, summary.json and review_summary.json retain the evidence. Prices and event tables were hashed before and after; all four source hashes are unchanged. Five synthetic identity tests check the formulas. Event counts reconcile to output classifications; source rows were not modified. This is a targeted event audit, not full-market independent validation.
'''
    (out/'README.md').write_text(report)
    # Verify delivered event files and review findings agree.
    assert len(pd.read_csv(out/'split_events.csv'))==sum(summary['splits_by_class'].values())
    assert len(pd.read_csv(out/'cash_events.csv',low_memory=False))==sum(summary['cash_by_class'].values())
    assert stats['material_scale_matches_50bp']<=stats['material_special_events']
    assert np.isfinite(material.relative_error).all()
    print(json.dumps(stats,indent=2));print('AFG checks');print(cr[cr.security.eq('AFG US Equity')&cr.special_date.ge('2023-01-01')].to_string(index=False));print('OUTPUT',out)

if __name__=='__main__':main()
