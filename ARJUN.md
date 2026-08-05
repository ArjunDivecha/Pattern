# ARJUN.md — product owner's memo

*Fable 5, 2026-07-06. Written to you, not to an agent.*

## What this repo is worth

This is a **successful, honest, research-complete** project — and the trading thesis inside it is
**closed**, not open. You asked two questions and the repo answered both: (1) *Does JKX 2023
replicate?* — yes, cleanly (LS Sharpe 1.22, NW t +7.99, +17.4%/yr, 25/28 years positive). (2) *Can I
trade it?* — mostly no. The alpha concentrates in small-cap, high-vol, recent-loser names; capacity is
~$500k–$1M/side; and README addenda H and I document four separate ETF / single-stock-ETF deployment
attempts that all failed for *structural* reasons (no issuer wraps the names that carry the edge). The
one survivor is a single R1000 cell (mid-cap × high-vol × recent-loser: ~+13% net CAGR, t 3.86).

So: **alive as a research asset and a reusable pipeline; dead as a standalone product.** The value
from here is not "trade Pattern" — it's harvesting the signal into other machinery and freezing the
lessons. One ecosystem flag: there is a **`pattern 2`** repo in your active list. Before investing
more here, confirm whether `pattern 2` is a fork/continuation that supersedes this — if so, consolidate
and mark one as canonical; you don't want two live copies of a chart-CNN pipeline drifting apart.

## Extensions ranked by value ÷ effort

1. **Harvest the negative result into personal-knowledge (10 min, highest ROI).** The most expensive
   thing you own here is the *lesson*: "chart-image CNN alpha is real but capacity-bound to illiquid
   small-caps and is not ETF-portable — proven, not conjectured." Write it once via
   `mcp__personal-knowledge__create_entry` so you (and your agents) stop re-running this experiment.
   First step: create an entry tagged to Pattern with the H/I structural-wall conclusion.

2. **Use `p_up` as a *feature*, not a *strategy*.** AUC is 0.507 (diffuse) but the portfolio t is ~8 —
   that profile is a textbook *factor input*, not a standalone book. Feed the ensemble
   `p_up_mean`/`p_up_std` from `runs/*/predictions.parquet` into **T2 Factor Timing** / **Fable Daily
   Trading** as one more cross-sectional signal, where diffuse-but-real edges belong and where trading
   costs are already handled at the sleeve level. First step: export the R1000 `predictions.parquet` to
   a monthly factor panel keyed by `(ticker, date)`.

3. **Freeze the leakage guard (P0 in FABLE.md).** The whole result rests on the split purge, and it's
   currently untested. This is the prerequisite for trusting any *reuse* of the pipeline. Nearly
   hands-off: the contract is authored and validated — hand it to **Codex** in Build Mode. First step:
   save the fenced block in FABLE.md to `pattern-leak-purge-guard.spec.md` and run the handoff prompt.

4. **Cost-realistic re-test of the one tradable cell via the QuantConnect/LEAN skill.** The mid-cap ×
   high-vol × recent-loser cell is the only survivor; the in-repo cost model is a Corwin-Schultz
   estimate. Reuse the **backtest** skill to price it against a real fill/impact model before you ever
   commit capital. First step: express the cell as a monthly rebalance rule on its ~71-name universe.

5. **Turn the pipeline into a portable "score any universe" tool (post-P0).** The durable asset is
   *"OHLCV panel → chart-CNN cross-sectional signal in one command."* You already have
   `src/build_r1000_database.py` / `build_nky_database.py` (Bloomberg-backed). Wrap build → train →
   `score_live` behind one CLI so a new universe is a config, not a project. Reuse: **OpusBloomberg** +
   the **bloomberg** skill. Higher effort (~1 day); only worth it if you'll run ≥2 more universes.

6. **Archive the dead deployment code.** `scripts/fetch_sse_data.py`, `finetune_etf_w11.py`,
   `backtest_sse*.py`, `render_etf_cache.py` back the H/I dead ends. Move them to `attic/` (don't
   delete — they're evidence) so the live surface reflects what's actually in use.

## Quick wins (< 1 hour)
- **Clean the working tree.** 14 modified files + 1 untracked (`scripts/plot_mid_cell_since2020.py`),
  ~653 insertions, sitting uncommitted. Review and commit or revert — right now the repo's "true" state
  is ambiguous and any agent building on it inherits your half-finished edits.
- **Log the lesson** (extension 1) — 10 minutes, saves days of recomputation.
- **Run the P0 contract** — mostly Codex's time, not yours.

## What NOT to do
**Do not spend another hour trying to make this a tradable ETF or single-stock-ETF product.** Addenda
H and I already proved, four different ways, that this fails *structurally*: the alpha lives in
small-cap high-vol losers, and on liquid/mega-cap names the signal is flat or inverted (SSE underlying
LS −3.6%/mo, t −2.14). That's a wall, not a regime you can wait out or a hyperparameter you can tune.
And don't push the R1000 cell past ~$1M/side — the capacity ceiling is the whole point of the cost
analysis. The tempting move is "one more deployment idea"; the disciplined move is to bank the signal
as a feature (extension 2) and move on.
