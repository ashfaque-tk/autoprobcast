# Project Status

Milestone: Upload VN2 → Interpret schema → Confirm mappings → Standardise data → Validate data → Automated EDA → Capability assessment.

Scope: the VN2 sales file only, decided 2026-11-19. The first goal is a probabilistic sales forecast. Inventory files are not uploaded.

## Completed
- Task 001 — Packaging: `uv_build` backend, `openpyxl` and `pytest` (dev) added.
- Task 001 — `load_dataset` in `data/ingestion.py`, with 12 passing unit tests.
- Task 001 — Streamlit upload and preview (`app.py`), checked manually with the VN2 sales file.
- Task 002 — Schema interpretation in `data/schema.py`, with 14 tests passing. On the VN2 sales file it gives a `wide` layout, maps `Store` and `Product`, and finds 157 date columns (2021-04-12 → 2024-04-08) at `W-MON`, with no unmapped columns.
- Task 003 — Confirm mappings: `confirm_mapping` with 10 tests, plus the UI. Checked manually in Streamlit.
- Task 004 — Standardisation: `standardize` with 7 tests. On VN2 it gives 94,043 rows, no NaN, dates 2021-04-12 → 2024-04-08. Streamlit preview checked manually.
- Task 005 — Validation: `validate` with 20 tests. On VN2: 94,043 rows, 599 series, frequency W-MON, no issues found, runs in about 1.2 s.
- Task 006 — EDA (2026-11-25): Claude wrote `aggregate_trend` and `seasonal_profile` at the user's request. All 76 tests pass. On VN2 the whole EDA takes 0.19 s, 52 seasonal positions, and 555 of 599 series have a `seasonal_acf`. The Streamlit wiring loads without errors (AppTest); the upload flow still needs a manual check. Charts now have captions (both are dataset-level: trend = sum over all series, seasonal = equal-weight mean of sales / series mean), and there is a single-series view: a (store, product) dropdown, its sales chart and its profile row.

## In progress
- Task 008 — `notebooks/feature_workbench.ipynb` (2026-11-27): only the VN2 load cell (copied from `eda_workbench`). The user builds features there in pandas / mlforecast first, to understand the logic; numpy or polars is decided afterwards.
- Task 008 — Feature engineering skeleton, revised 2026-11-25 after the VN2 winner's report (Szabłowski 2026) and the frequency review:
  - `src/demand_platform/frequency.py` (new): an explicit `FrequencyProfile` per frequency (W, D, M/ME/MS) holding every window and lag: seasons, lags, rolling, EWM, shape, trend, last-year half-width, difference lags, spike min. observations, Fourier orders. Unsupported → None, and seasonal behaviour is switched off. Only weekly is checked against data. Daily year = 364 (same weekday). EDA now takes its season and trend window from it (`SEASONAL_PERIODS`, `TREND_WINDOW` removed; weekly results unchanged; trend_ratio is now NaN for unsupported frequencies). The app's frequency wording uses the profile.
  - `forecasting/features.py`: 8 groups (lags, rolling, shape, trend, intermittency, seasonal, calendar, ids), all derived from the profile. User bodies: `nonzero_rate`, `periods_since_last_sale`, `periods_since_robust_spike` (median/MAD, ROBUST_Z = 3.5), `difference` (the report's momentum; slope = Δk/k is redundant for trees), `seasonal_correlation` (causal expanding Pearson = the report's seasonality strength), `fourier(dates, season, order, kind)` (calendar-based, year/week). `build_forecaster` takes `target_transforms`; the docstrings require `fit(..., dropna=False)`.
  - `forecasting/scaling.py` (new): `AnnualScale` mlforecast target transform = max(year × mean of last year, 1), warm start (expanding mean) below 85 % of a year; fixed per series at fit time (the report recomputes it every week). User writes `fit_transform` and `inverse_transform`; `scale_for(ids)` is there for the sampling callback.
  - `forecasting/dataset.py`: `recency_weight` (user), yearly blocks per series, decay 0.5.
  - Tests: `test_frequency.py` (5, pass), `test_features.py` (22), `test_scaling.py` (5), `test_forecast_dataset.py` (6). 22 fail with NotImplementedError until the bodies are written; all 123 tests pass against a reference implementation kept outside the repo.
  - VN2 check with the reference bodies: 40 features, preprocess 0.6 s, LightGBM fit 3.5 s, one recursive 3-week predict 1.25 s. The intermittency group costs ~1.1 s of that (Python per-series transforms, the sliding-window median the most expensive); built-ins alone take 0.05 s. At 1,000 paths that is ~20 min per origin → for Task 010: numba, a cheaper spike rule, or computing intermittency features only once per path batch.
  - Not adopted from the report (yet): stockout masking (needs In Stock data), median imputation (ablation candidate), dynamic weekly-updated scale, Optuna tuning, direct strategy (comparison only). Horizon for VN2 = 3 (lead time 2 + review 1); competition test = origins Week 0…5, actuals from Week 8.
- Task 007 — Capability assessment (2026-11-25): `analytics/capability.py` is done. The user wrote `history_capability`; Claude fixed it (missing `.mean()`, `>` changed to `>=`, the key `share_two_cycle` changed to `share_two_cycles`, a dead empty branch removed) and wrote `seasonality_capability` and `distribution_capability` at the user's request. All 86 tests pass. On VN2: 599 series, none never sold, median 156 periods, 93% have ≥ 2 cycles, 555 are seasonal-eligible with a median seasonal ACF of 0.13 (35% are above 2/√n), count data, a pooled zero share of 0.385, median ADI 1.75, and 97% overdispersed. The UI still needs a manual check.
- Task 006 — `notebooks/eda_workbench.ipynb` is minimal: it loads `vn2`, `toy1` (data problems: late start, NaN, never sold, a return) and `toy2` (shapes: constant, single row, spike, long zero run). `ipykernel` was added (dev), and `polar` was removed (an unrelated package, not Polars).
- Task 006 — `series_profile` is done: the user's draft was reviewed and corrected. All 8 profile tests pass, it matches an independent reference on toy1, toy2 and VN2, and it runs in 0.18 s on VN2 and 9.7 s on 4.7M rows / 30k series. `aggregate_trend` and `seasonal_profile` are still to be written.
- Task 006 — EDA: the user designs and writes the helpers in their own way. `eda.py` holds only the three public functions; their docstrings are the contract. Tests cover only the public functions and `_seasonal_period` (12 fail until written). Claude reviews and adds helper tests afterwards.

## Pending
- Open design question: frequency-aware seasonality and trend window (daily data has weekly *and* yearly seasonality; `TREND_WINDOW = 13` is in rows). Proposed, not approved.

## Decisions
- App layout (2026-11-25, revised): the sidebar holds upload, column mapping and "Confirm mapping"; after confirming, everything runs automatically (cached with `st.cache_data`). The main view is the sales chart at a chosen level (store × product, store, product, all series); forecasts will be drawn on it later. Below it is a short plain-language summary (frequency as words such as "Weekly data (dates on Mondays)", history in weeks or years, seasonality peak, sales values, data checks). There are no metric tiles or jargon such as "count data" or "seasonal cycles" on the main view. Everything else sits in one "Details" expander with tabs.
- Ingestion is a pure function, `load_dataset(file, filename) -> DataFrame`, with no Streamlit import and no disk paths.
- Ingestion does no interpretation. Every column is read as `dtype=str`, and type conversion belongs to standardisation.
- `IngestionError` is the only exception the UI catches. Anything else is a bug and should surface.
- A header-only or empty file is rejected.
- Missing cells stay NaN (no `keep_default_na=False`), so validation and EDA can see them.
- Excel: only the first sheet is read. Multiple sheets are not concatenated, because their schemas may differ.
- `workflow.py` stays empty until there is more than one pipeline step.
- Schema interpretation only *proposes* a mapping (`SchemaProposal`). The user confirms it, and it never transforms data.
- Both wide (dates in headers) and long (a date column) layouts are supported. Anything else gets layout `"unknown"`.
- Dates are parsed with the strict format `%Y-%m-%d`. `format="ISO8601"` was rejected because it parses `"2021"` and `"2021.5"` as dates, so numeric IDs would be misread.
- Roles are guessed by exact synonym match. If two columns match one role, neither is mapped and the user decides.
- `infer_freq` runs inside `pd.option_context("future.infer_freq_returns_offset", True)`. A global `set_option` was replaced, because it changed pandas for the whole process and removing it outright brings back a pandas FutureWarning.
- Standardisation: `standardize(df, schema) -> DataFrame` with the canonical columns in order, sorted by `store_id, product_id, date`. IDs stay `str`, `date` is datetime, and `sales` is float (`"3.0"`, NaN allowed). Unparseable values become NaN or NaT, and no rows are ever dropped or filled, so validation can report them.
- Canonical columns: `store_id`, `product_id`, `date`, `sales`.
- A confirmed mapping is a `SchemaProposal` (built with `dataclasses.replace`), not a new type. It is stored in `st.session_state["confirmed_schema"]` and cleared when a different file is uploaded.
- `confirm_mapping` requires both `store_id` and `product_id`, and raises `ValueError` for: an unknown layout, a missing role, one column used for two roles, a date column or a non-existent column picked, a long layout without a target, or a wide layout with a target.
- Validation: `validate(df, frequency) -> ValidationReport`, a frozen dataclass that holds counts only and changes no data. `is_valid` is False only for blocking issues: duplicate keys (NaT excluded, `keep=False`), NaT dates and ±inf sales. NaN sales, negative or non-integer sales, off-grid dates and gaps inside a series are warnings. Gaps are counted only within each series' own first→last span. With `frequency=None` the calendar checks are skipped.
- Roadmap (2026-11-20): finish Tasks 006–007, then stop and plan. After that: a full VN2 forecast, then a test on a second dataset (M5 or other), then inventory data with AI column identification. The goal is fully probabilistic, sample-path forecasts, with the output format still to be agreed. Every task is skeleton + tests, with the user writing the bodies.
- EDA: one row per series, computed over its active span (from the first positive sale). It measures evidence for each family of model features (seasonal ACF, lag-1 ACF, trend ratio, zero runs, spikes, group mix). The features themselves are computed later, using past data only. Seasonal periods come from a fixed table (W→52, D→7, M→12), with no guessing, and need ≥ 2 cycles. There are no ABC or demand classes: the plan is a single global probabilistic model, so EDA informs dataset-level decisions (features, likelihood, scaling, training window). `dispersion` = var/mean feeds the choice of sampling distribution. NaN is unknown, never zero.
- EDA series keys (2026-11-24): `SERIES_COLUMNS` is named once, and `groupby(...).ngroup()` turns it into one integer series id that every later groupby uses. Hierarchy columns (M5 `dept_id`, `cat_id`) are attributes of a series, not part of its key.
- EDA is vectorized (2026-11-23): no per-series loop. Lags and windows count rows, not calendar periods (VN2 has no gaps).
- Polars (2026-11-23): stay on pandas for this milestone. At the Task 007 stop, benchmark validate and EDA on M5 (time and peak memory), and consider `category` IDs. If it's still too slow, migrate everything from `standardize` onward in one task. Never mix pandas and Polars.

## Open issues
- `.xls` is in `SUPPORTED_EXTENSIONS`, but `xlrd` isn't installed. A real `.xls` upload raises an uncaught `ImportError`. Either add `xlrd` or remove `.xls`.
- Still open: should long-layout date detection tolerate a few unparseable values?

## Next task
Milestone 1 (tasks 001–007) is done. Plan for milestone 2, agreed 2026-11-25: forecast with the data we already have (sales only), then measure accuracy. Extra data (inventory, price, calendar, hierarchy) comes later through tagged multi-file upload, because we don't know what users can provide.

Evaluation data (corrected 2026-11-25): the forecast origin is Week 0 (2024-04-08), and the target is the next weeks (typically 4). Each `Week N` file is cumulative: it holds exactly one more week than `Week N-1`, and the history is identical (checked for Week 8 against Week 0). `Week 1` therefore holds only 2024-04-15; the next 4 weeks are complete in `Week 4`. Plan: develop on rolling origins inside Week 0, then test once on the competition weeks using `Week 8` as the actuals source (origins Week 0…4 with H=4, like the weekly competition).

Tooling decided (2026-11-25):
- `mlforecast` 1.0.2 + `lightgbm` 4.7.0 added (user-approved). mlforecast handles training, leak-safe recursive features and backtest windows; it has no sample paths (conformal intervals only).
- Sample paths are custom: recursive `predict(h, after_predict_callback=draw)`, where `draw` replaces the predicted mean with a random draw from the chosen distribution. Verified in the mlforecast source (`core._predict_recursive`: callback → `_update_y`) and by a test on VN2: the forced value appears in the next step's `lag1`, `lag2` and rolling mean, and `predict` returns the draws. The callback receives a NumPy array. Caveat: with `target_transforms` the callback sees transformed values. Direct (`max_horizon`) has no callback and gives no joint paths; it serves only as a per-week comparison.
- Cost on VN2 (599 series, LightGBM Poisson, 200 trees): fit 0.7 s, about 38 ms per 4-week path, so 1,000 paths ≈ 40 s per origin. For larger data, either replicate each series N times in one `predict` call (fast, N× memory) or loop over paths (light memory, time linear in N); decide in Task 010.
- Sample-path format: a NumPy array `(n_series, n_paths, horizon)` of int32, plus a series-key frame and a date index. A long table (one row per series × date × path) is ruled out: M5 at 1,000 paths × 28 days would be about 850M rows.
- Feature selection: feature groups + ablation on CRPS/MASE over development origins; SHAP only for understanding. Statics (e.g. seasonality strength) are computed from history up to the origin only.

### Task 008 — Feature engineering (sales only)
- First, the forecast contract: origin (last known week), horizon H, and a split `history ≤ origin`, `actuals origin+1 … origin+H`. The features depend on it, because "past only" means past relative to the origin.
- Approach: **recursive**. A one-step-ahead model; sample paths are made by feeding each sampled week back in as history. Direct (one model per horizon) only gives each week's distribution separately, not joint paths, so sums over weeks would be wrong. Consequence: every feature must be recomputable from history + simulated values, using values ≤ t−1 only.
- Candidate features: lags (1–4, 52 where it exists), rolling mean / std / zero share over 4, 13, 26 weeks (shifted by one), weeks since last sale, series age, calendar (ISO week of year, month), series level (expanding mean to t−1).
- Leakage test: changing sales at t or later never changes features at t.
- Split: Claude writes the skeleton + tests; the user writes the feature bodies.

### Task 009 — Metrics and baselines
- Point metrics: MAE, RMSE, bias, MASE (scaled by in-sample naive error).
- Probabilistic metrics from sample paths: CRPS, pinball loss at chosen levels, interval coverage and width (e.g. 50 / 80 / 95 %).
- Levels: store × product, store, total. Aggregates come from summing sample paths, which keeps them consistent.
- Baselines producing sample paths in the same format: empirical bootstrap of recent weeks; Poisson / negative binomial around a moving average. The model must beat these.
- Decide the sample-path output format here (shape and storage).

### Task 010 — First global probabilistic model
- Recommended: LightGBM (new dependency: a fast global tree model that handles non-linear features and series IDs) for the one-step mean with a Poisson/Tweedie objective; negative-binomial dispersion fitted from residuals; recursive simulation of N paths. Alternatives: an NB GLM (simpler, linear) or DeepAR-style neural nets (heavier, later).
- Rolling-origin backtest against the baselines, then one run on the final holdout.

### Task 011 — Forecast in the app
- A fan chart (median + intervals) on the main sales chart at every level, plus a short accuracy summary.

### Open decisions (to settle at the start of Task 008)
- Horizon H and the number of rolling origins.
- The number of sample paths.
- Recursive vs. direct (recommended: recursive).
- The model library (recommended: LightGBM).

### Later (after the first forecast)
- Multi-file upload with tags (sales, inventory/stock, price, calendar, hierarchy): VN2 `In Stock.csv` (stockouts censor sales) and `Master.csv` (product/store hierarchy).
- Frequency-aware seasonality and trend window; `share_seasonal_significant` in `CapabilityReport`; the M5 benchmark (Polars decision); tabs/levels for hierarchy.
