# BRFSS Data Preprocessing — FSE570 Capstone

## Data Source
CDC Behavioral Risk Factor Surveillance System (BRFSS)
Years: 2020–2024

---

## Files in This Folder

| File | Description |
|------|-------------|
| `brfss_preprocess.ipynb` | Main preprocessing pipeline |
| `brfss_group_summary.csv` | Weighted obesity rates by demographic cell — main input for MrP model |
| `brfss_sparse_cells.csv` | Cells with fewer than 30 respondents — flagged for model smoothing |
| `brfss_state_estimates.csv` | Weighted state-level obesity estimates (all years combined) |
| `brfss_state_year_estimates.csv` | Weighted state-level obesity estimates by year |
| `brfss_state_obesity_rates.png` | Bar chart of weighted state obesity rates |

---

## What the Preprocessing Notebook Does

### Step 1 — Load raw XPT files
Five years of BRFSS data loaded from CDC SAS transport format files. Raw shapes:
- 2020: 401,958 rows × 279 columns
- 2021: 438,693 rows × 303 columns
- 2022: 445,132 rows × 328 columns
- 2023: 433,323 rows × 350 columns
- 2024: 457,670 rows × 301 columns

### Step 2 — Select variables
From ~300 raw columns, we keep only what's needed for modeling:
`_STATE`, `_LLCPWT`, `_BMI5`, `_AGEG5YR`, `_SEX`, `_EDUCAG`, `_INCOMG1`, `_RACEPRV`

### Step 3 — Handle race variable
**Problem:** CDC renamed and restructured the race variable in 2022, making cross-year comparison unreliable.

**Variable chosen: `_RACEPRV`** (and its 2022 equivalent `_RACEPR1`)

Reasons:
1. `_RACEPRV` is derived from both `_RACE` and `_IMPRACE` — it fills in missing race values through imputation, reducing missing data from 2–3.2% per year to 0%
2. In 2022, CDC renamed `_RACE` to `_RACE1` and `_RACEPRV` to `_RACEPR1` — same variable, just renamed. Using `_RACEPRV`/`_RACEPR1` gives the most consistent cross-year variable
3. `_RACEPRV` is CDC's recommended variable for internet prevalence tables per the calculated variables documentation

**Imputation analysis:** Of the 8,987–14,055 rows with missing raw race per year, 83–93% get assigned NH-White by the imputation. This slightly inflates NH-White counts — the effect is at most +0.9% of total rows in any year.

**2022 harmonization:** In 2022, `_RACEPR1` combines Other and Multiracial into a single code 6. We apply the same collapse to all other years for consistency:
- Other (code 6) and Multiracial (code 7) → merged into Other/Multiracial (code 6)
- Hispanic (code 8) → recoded to 7

Final race categories (codes 1–7, consistent across all years):

| Code | Label |
|------|-------|
| 1 | NH-White |
| 2 | NH-Black |
| 3 | AIAN |
| 4 | Asian |
| 5 | NHOPI |
| 6 | Other/Multiracial |
| 7 | Hispanic |

### Step 4 — Handle income variable
**Problem:** 2020 used `_INCOMG` (5 categories, max 50k+) while 2021–2024 use `_INCOMG1` (7 categories, up to 200k+). These are not comparable — 2020's code 5 captures everyone earning 50k+ while later years split that into three separate bins (50k–100k, 100k–200k, 200k+).

**Decision: Option A — Set 2020 income to NaN**

Two options were evaluated:
- **Option A:** Drop 2020 income (set to NaN), use 7-bin detail for 2021–2024
- **Option B:** Collapse all years to 5 bins (50k+)

Simulation results comparing obesity rate variation across income bins:
- Option A (7 bins, 2021–2024): std dev = 0.051, range = 0.147
- Option B (5 bins, all years): std dev = 0.023, range = 0.059

Option A preserves more than twice the variation in obesity rates across income groups. Collapsing to 5 bins would significantly weaken income as a model predictor.

Groups most affected by losing high-income granularity (weighted % earning 50k+):
- College grads: 82.6%
- Asian: 70.5%
- NH-White: 66.3%
- Ages 40–54: 67–69%

**Impact:** 2020 rows are kept for all non-income analyses. Income-stratified analyses use 2021–2024 only (1,322,240 rows).

Final income categories:

| Code | Label |
|------|-------|
| 1 | <15k |
| 2 | 15k-25k |
| 3 | 25k-35k |
| 4 | 35k-50k |
| 5 | 50k-100k |
| 6 | 100k-200k |
| 7 | 200k+ |
| NaN | 2020 (not comparable) |

### Step 5 — BMI filtering and obesity indicator
Invalid BMI codes removed before calculation:
- `_BMI5 ≥ 9000` = missing/refused codes
- `_BMI5 ≤ 1200` = implausible values (BMI ≤ 12)

BMI calculated as `_BMI5 / 100`. Obesity indicator: `obese = 1 if BMI ≥ 30`.

### Step 6 — Filter missing demographics
Rows with missing/refused values removed for age, sex, education, race. 2020 income excluded from filter since it is intentionally NaN.

### Step 7 — Combine and save
All 5 years concatenated into a single file. Final shape: **1,622,499 rows × 11 columns**.

---

## Variables in Clean Dataset

| Variable | Description |
|----------|-------------|
| `_STATE` | State FIPS code (1–78, includes territories) |
| `_LLCPWT` | Survey weight — must be used for all population-representative estimates |
| `_BMI5` | Raw BMI × 100 (e.g. 2500 = BMI 25.00) |
| `BMI` | Calculated BMI (`_BMI5 / 100`) |
| `obese` | Binary obesity indicator (1 if BMI ≥ 30, else 0) |
| `_AGEG5YR` | Age group code (1–13) |
| `_SEX` | Sex (1=Male, 2=Female) |
| `_EDUCAG` | Education level (1–4) |
| `_INCOMG1` | Income group (1–7, NaN for 2020) |
| `_RACEPRV` | Race/ethnicity (1–7, harmonized across all years) |
| `year` | Survey year (2020–2024) |

---

## Demographic Bins
These bins must match exactly with the ACS team's population distributions.

**Age:** 18-24, 25-29, 30-34, 35-39, 40-44, 45-49, 50-54, 55-59, 60-64, 65-69, 70-74, 75-79, 80+

**Sex:** Male, Female

**Education:** Did not graduate high school, Graduated high school, Attended college or technical school, Graduated college or technical school

**Income:** <15k, 15k-25k, 25k-35k, 35k-50k, 50k-100k, 100k-200k, 200k+ (2021–2024 only)

**Race:** NH-White, NH-Black, AIAN, Asian, NHOPI, Other/Multiracial, Hispanic

---

## Group Summary Output

`brfss_group_summary.csv` contains weighted obesity rates for each demographic cell. Each row is a unique combination of age × sex × education × income × race.

| Column | Description |
|--------|-------------|
| `age_group` | Age group label |
| `sex` | Sex label |
| `education` | Education label |
| `income_group` | Income label |
| `race_group` | Race label |
| `obesity_rate` | Weighted obesity rate using `_LLCPWT` |
| `n` | Number of BRFSS respondents in this cell |
| `reliable` | 1 if n ≥ 30, 0 if too sparse |

Total cells: 4,953
Reliable cells (n ≥ 30): 2,893 (58.4%)
Sparse cells (n < 30): 2,060 (41.6%)

---

## Sparse Cell Analysis

Of 4,953 demographic cells, 41.6% have fewer than 30 respondents. These cells have unreliable obesity rate estimates.

Most affected race groups:
- NHOPI: 623 sparse cells
- Asian: 431 sparse cells
- AIAN: 419 sparse cells

Most affected income groups:
- 200k+: 446 sparse cells
- <15k: 335 sparse cells

The `reliable` column in `brfss_group_summary.csv` flags cells with n ≥ 30. The modeling team should apply partial pooling or multilevel smoothing for sparse cells. Consider collapsing NHOPI into Other/Multiracial for modeling given extreme sparsity.

---

## State-Level Results

Weighted obesity estimates by state serve as a validation baseline — county estimates should be consistent with their state averages.

**Highest obesity states (2020–2024):**
1. West Virginia — 41.7%
2. Mississippi — 41.4%
3. Alabama — 40.3%

**Lowest obesity states:**
- Colorado — 25.5%
- Hawaii — 26.2%
- District of Columbia — 24.4%

**National weighted average: 33.9%**

These match known CDC obesity rankings, validating the preprocessing pipeline.

**Trend:** Obesity rate increased from 32.9% in 2020 to 34.2% in 2024, with a notable jump between 2020 and 2021 likely reflecting COVID-19 behavioral changes.

---

## Survey Weights
All obesity rate calculations use `_LLCPWT` survey weights. BRFSS uses complex sampling — different respondents represent different numbers of people in the population. Ignoring weights produces biased estimates.

The `obesity_rate` column in `brfss_group_summary.csv` is a weighted mean. The `n` column is unweighted count for sample size reference.
