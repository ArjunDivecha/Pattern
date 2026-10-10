# Clean research overlay v1

Build: `.venv/bin/python research/clean_universe/build.py`.
Output: `data/research_universe/clean_I20_R5_v1/`.

The source frozen database and both image shards remain untouched. Join eligibility by `cache_shard,row_id` or unique `security,date`; row IDs alone are not globally unique.

For upcoming I20/MA20/R5 studies, select formation_eligible for inference and training_eligible for fitting; use next_open_return_5_clean for training. The existing legacy trainers are NOT automatically migrated to this overlay and must be wired to it explicitly before launching new fits. Do not use the image index's old close-to-close return as the training label.

Formation eligibility uses the signal-date adjusted close >=5 and no absolute one-day close or TRI return above100% within the39-bar effective image history. Existing cache numerical and calendar validity remain prerequisites. This is a conservative adjusted-price proxy, not a certified historical nominal-price floor. Extreme moves are suspect, not proven errors; no prices are repaired or winsorized.

Label validity requires continuous sessions and valid bars/endpoints through t+5. Return is close(t+1)/open(t+1) times TRI(t+5)/TRI(t+1), minus1. Returns outside[-100%,100%] are retained raw, flagged and excluded from training. Real large losses within that interval are retained. Missing future dates remain missing.

Evaluation must form portfolios using formation_eligible alone, retain raw returns, and report suspect/missing outcomes. Outcome filtering is only a separately labelled diagnostic, never a reason to remove a holding silently. Quarantining training labels does not establish a corrected economic return. Any later resolution requires source evidence.
