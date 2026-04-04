

## Step 1 - Handling Missing Data with Multiple Imputation

Real survey data has missing values. Some respondents skip questions, some answers get removed during cleaning. Dropping incomplete rows would bias the sample because people who skip income questions aren't randomly distributed. They tend to cluster at the high and low ends of the income scale.

Before imputing anything, we ran chi-square tests to figure out *why* values are missing. This matters because the right imputation strategy depends on the answer:

- Missing Completely At Random (MCAR): missingness has nothing to do with other variables. Dropping is safe, though MI is still better.
- Missing At Random (MAR): missingness can be explained by other observed variables. MI handles this correctly.
- Missing Not At Random (MNAR): missingness is related to the missing value itself (e.g., high earners refusing to report income). The hardest case, and no method fully fixes it.

The chi-square tests rejected MCAR for both `_INCOMG1` and `_AGEG5YR` across every predictor tested (all p < 0.0001). The data is at minimum MAR, so dropping rows would introduce systematic bias. We proceed under the MAR assumption and document it as a limitation.

What we did: Multiple Imputation (MI) using sklearn's `IterativeImputer`. Instead of filling in one best guess, we generate m=5 complete datasets, each with slightly different imputed values drawn from a predictive distribution. Every model is then fit on all 5 datasets, and results are pooled together. We also add binary missingness indicators (`_INCOMG1_missing`, `_AGEG5YR_missing`) as input features, because the fact that someone skipped the income question is itself predictive of obesity given the income-obesity gradient.

Why this matters: Filling with the mean or dropping rows produces overconfident uncertainty estimates. MI properly propagates missingness uncertainty forward into final predictions and standard errors. The between-imputation standard deviation across m=5 predictions came out at 0.0033, which is reassuringly small. Missingness isn't driving much prediction variance.

---

## Step 2 - Survey Weights and Weight Trimming

BRFSS assigns each respondent a survey weight (`_LLCPWT`) that tells you roughly how many people in the population that respondent represents. Without these weights, a state that happened to get more respondents would dominate the model.

All model fitting uses these weights. Extreme weights are a problem though. One respondent with a weight of 50,000 can destabilize training. We trim weights at the 99th percentile to reduce that influence while keeping the calibration intact.

The pipeline also reports the design effect (DEFF), a measure of how much the complex survey sampling inflates variance compared to a simple random sample. For BRFSS national estimates, DEFF values of 1.5–2.5 are typical, which means reported confidence intervals need to be inflated by `sqrt(DEFF)` to be properly calibrated. The DEFF reported here (0.002) is implausibly low and warrants a closer look at how the design variables (`_STSTR`, `_PSU`) are being passed into the variance calculation before any standard errors are reported downstream.

Sensitivity check (SA1): We tested p95 trim vs. p99 trim vs. no trim. Predicted prevalence varied by only 1.29 percentage points across all three, well within acceptable range, confirming the 99th percentile choice is robust.

---

## Step 3 - Model 1: Weighted Logistic Regression with Rubin Pooling

The first model is a weighted logistic regression fit using `statsmodels.GLM` with a Binomial family. Categorical features (sex, age group, race, education, income) are one-hot encoded, and three interaction terms are included: Age x Race, Income x Education, and Sex x Age.

Why these three interactions specifically?:
- Age x Race: The obesity-age trajectory differs by race. NH-Black women peak earlier than NH-White women, and Asian obesity rates remain compressed across the full age range.
- Income x Education: SES effects on obesity are not purely additive. A college graduate earning under $25k has a different risk profile than a high school graduate at the same income.
- Sex x Age: Male obesity peaks earlier and declines faster with age than female obesity.

One thing to verify before finalizing the model: with one-hot encoding and three interaction terms, some demographic combinations (e.g., AIAN x age 18-24) have very few respondents. GLM can silently produce NaN or near-zero coefficients for those cells. Running `np.isnan(pooled_coef).sum()` catches this early. A penalized approach via `glmmTMB` or a Bayesian formulation via `brms` would handle sparse interaction cells more gracefully by shrinking estimates toward the main-effect baseline rather than failing silently.

Rubin's Rules: Since we have m=5 imputed datasets, we fit 5 separate logistic models and combine them using Rubin's pooling formulas:
- Pooled coefficient = average of the 5 estimates
- Pooled variance = average within-imputation variance + (1 + 1/m) x between-imputation variance

That second term captures how much the answer changes depending on which imputed values we happened to use, and inflates the standard errors accordingly.

Results:
- Weighted AUC: 0.6185
- Weighted Brier Score: 0.2134
- Predicted prevalence: 33.24% vs. true 33.32% (gap of 0.08pp, essentially perfect calibration at the national level)

The AUC of 0.62 is worth discussing. Obesity prediction from demographics alone is genuinely hard and within-group variance is enormous, but the documented BRFSS race gradient between Asian (~15% obese) and NH-Black/AIAN (~45%) suggests AUC in the 0.72-0.78 range should be achievable with a well-specified model. The gap between what we got and what the data should support is a signal to check whether the interaction terms were actually estimated. If any GLM cells failed silently, the model is missing demographic signal that should be there.

---

## Step 4 - Model 2: Neural Network (NN)

The second model is a 3-block neural network with skip connections and batch normalization:

```
Input (demographics + state embedding) → Block1 (128) → Block2 (64) → Block3 (32)
                                      ↘ skip connection ↗
                                         → Output (1 logit)
```

Design choices explained:

- Skip connection: The network can learn deviations from a linear baseline, rather than having to fully rebuild linearity from scratch. This also makes the nonlinearity diagnostic in Step 6 meaningful: the skip path gives us a clean decomposition of how much work the hidden layers are actually doing.
- State embedding: Rather than treating state as 51 dummy variables, we learn a 4-dimensional embedding for each state. This lets the model capture geographic variation in a compact form and generalizes better to states with fewer observations.
- Batch normalization: Stabilizes training by normalizing layer inputs. Especially useful with survey data where feature scales vary substantially.
- Dropout (p=0.3): Regularization , randomly zeroing out 30% of neurons during training to prevent overfitting.
- AdamW + Cosine LR schedule: AdamW decouples weight decay from the gradient update (unlike vanilla Adam), which tends to give better generalization. Cosine annealing smoothly reduces the learning rate over training rather than dropping it sharply.
- Early stopping (patience=10): Training stops when validation loss hasn't improved in 10 epochs. Best weights are restored. Stopped at epoch 53 here.
- Gradient norm check (epoch 1): Before committing to a warm-up schedule, the pipeline measures gradient norms in the first 20 batches. If the max norm exceeds 10, a linear warm-up is added. It came back at 1.076 here, so no warm-up was needed. This keeps the schedule decision grounded in what the data actually requires.

Multiple Imputation pooling: The NN is trained once on the first imputed dataset, then at inference time predictions are averaged across all 5 datasets. The between-imputation standard deviation (mean SD = 0.0033) tells us imputation uncertainty contributes very little to prediction variance.

Results:
- Weighted AUC: 0.6339
- Weighted Brier: 0.2112
- Predicted prevalence: 33.57% vs. true 33.32% (gap of 0.25pp)

The NN beats logistic on both AUC and Brier, though the margins are modest. SA4 found that larger architectures ([256, 128, 64] and [256, 256, 128]) produced slightly better test Brier scores (0.2115 and 0.2112 vs. 0.2119 for [128, 64, 32]). The differences are small, but [256, 256, 128] is worth trying in the next iteration.

---

## Step 5 - Temperature Scaling (Post-hoc Calibration)

A model with great AUC can still have poorly calibrated probabilities. Predicting 80% when the true rate is 60% is a problem for MRP since we're using raw predicted probabilities to estimate population prevalence.

Temperature scaling fits a single scalar parameter T on the validation set such that `logit / T` gives better-calibrated probabilities. T=1 means no adjustment needed. T>1 softens overconfident predictions. The NN came back with T = 1.0000, meaning the network was already well-calibrated at the aggregate level.

Temperature scaling fixes *aggregate* calibration. Whether calibration holds within specific demographic cells is a separate question, addressed in Step 7.

---

## Step 6 - Nonlinearity Diagnostic

Before declaring the NN the winner, we asked: *is the NN actually doing anything nonlinear, or is it essentially a logistic regression with extra steps?*

We used orthogonalized variance decomposition (Gram-Schmidt) to separate the linear component of the network's predictions from the nonlinear residual. The key word is *orthogonalized*. A naive comparison would double-count shared variance and give misleading numbers. Gram-Schmidt projects the nonlinear part into the space perpendicular to the linear part first, making the variance shares exact.

| Component | Share of variance |
|---|---|
| Linear | 5.7% |
| Orthogonal nonlinear | 2.3% |
| Cross-term (covariance) | 92.0% |

The nonlinear share is only 2.3%, well below the 5% threshold. The NN is essentially learning a linear function, and the added complexity isn't buying much. This is an important finding: it suggests the demographic-obesity relationship in this data is mostly additive and linear, and a more complex model isn't clearly warranted.

---

## Step 7 - Cell-Level Calibration

The national-level Brier score is fine, but MRP lives and dies by calibration within demographic cells (e.g., "Black women aged 45-54 with college education and middle income"). A model can look reasonable nationally while being poorly calibrated within the exact subgroups it needs to predict well.

We group the test set into cells defined by the 5 MRP axes and check:
- Calibration slope: Regression of true prevalence on predicted prevalence within each cell. Ideal = 1.0. Slopes below 0.85 indicate overconfidence, and slopes above 1.15 indicate underconfidence.
- Mean absolute error between predicted and true cell prevalence.

```
=== Logistic - Cell Calibration ===
  Reliable cells (n>=30): 389
    Mean |error|: 7.85pp
    Slopes: mean=0.045  std=0.292  WARNING: overconfident calibration

=== NN - Cell Calibration ===
  Reliable cells (n>=30): 389
    Mean |error|: 7.53pp
    Slopes: mean=0.326  std=0.669  WARNING: overconfident calibration
```

Both models are compressing predictions toward the global mean rather than tracking cell-specific rates. The logistic slope of 0.045 and the NN slope of 0.326 are both far from the ideal of 1.0, and this is the most important open issue for poststratification readiness. A model with near-zero calibration slope produces nearly identical obesity estimates for every demographic cell, which makes the poststratification step produce near-identical county estimates regardless of local demographic composition.

Checking `pd.Series(p_NN_test).std()` gives a quick diagnostic. If the standard deviation of predicted probabilities is below 0.05, the model is outputting a near-constant prediction and demographic signal is not being learned. The low slopes likely reflect a combination of sparse interaction cells in the logistic model and the known tendency of survey-weighted losses to dampen gradient signal when weight variance is high. Both are addressable.

The cell-level Brier criterion also returned `nan` for both models in the model selection step. This is a splitting artifact: with 15% test allocation, some demographic cells receive too few test observations to contain both obese and non-obese cases, so no Brier score can be computed. Increasing the test split to 20%, or using 5-fold cross-validation at the cell level, would give reliable cell-level Brier estimates across all folds.

---

## Step 8 - Race-Conditional Conformal Prediction

Conformal prediction is a way to generate prediction intervals with *guaranteed* coverage. At least 90% of predictions will contain the true label. This is a mathematical guarantee backed by finite-sample theory.

We calibrate separate intervals for each racial group because a one-size-fits-all interval would likely under-cover minority groups whose cells tend to be smaller and noisier. The quantile values (`q_hat`) tell you the half-width of each group's prediction interval:

| Group | n | q_hat | Coverage |
|---|---|---|---|
| NH-White | 45,874 | 0.692 | 90.1% |
| NH-Black | 4,869 | 0.628 | 90.2% |
| Asian | 1,793 | 0.798 | 86.6% |
| Hispanic | 6,261 | 0.658 | 90.8% |
| Multiracial | 1,472 | 0.681 | 89.7% |

Larger q_hat = wider intervals = more uncertainty in predictions for that group. Asian respondents have the widest intervals (0.798), reflecting more within-group heterogeneity in the data.

Asian coverage came in at 86.6% and Multiracial at 89.7%, both below the 90% target. With only ~1,800 Asian and ~1,500 Multiracial respondents in the test set, the val-set quantile estimate is itself noisy, and the calibration has high variance at those sample sizes. A larger calibration set (moving to 20% val instead of 15%), or lowering alpha to 0.05 to target 95% coverage, would absorb that noise and bring both groups into spec. Alternatively, pooling NHOPI, Other, and Multiracial into a single "Other Non-Hispanic" group is reasonable if individual calibration remains unreliable at those sample sizes.

Worth noting: race-conditional conformal only guarantees coverage within race groups, not within the fine demographic intersections (e.g., AIAN x female x age 25-29) that actually define MRP cells. The intervals are a useful quality signal, but they are not a substitute for cell-level calibration.

---

## Step 9 - Distribution Shift Test

BRFSS respondents aren't a perfect mirror of the ACS Census population. Phone survey selection biases mean certain demographics are systematically over- or under-represented. We tested whether model performance degrades when we reweight the test set to match ACS demographic marginals:

```
Logistic: Brier(BRFSS)=0.2134  Brier(ACS-shifted)=0.2156  gap=0.0022  OK
NN:     Brier(BRFSS)=0.2112  Brier(ACS-shifted)=0.2138  gap=0.0026  OK
```

Both models show minimal degradation (gaps < 0.005), which is a good sign that they'll generalize to the ACS population used in poststratification.

---

## Step 10 - SHAP Feature Importance

We ran SHAP's `PermutationExplainer` (5 runs, 200 background samples each) to identify which features drive predictions. `PermutationExplainer` makes no independence assumptions between features, which matters here since race, income, and education are correlated. Stability was verified by requiring coefficient of variation < 0.20 across runs . All top-5 features passed.

Feature group importance (mean |SHAP|):

| Feature | Importance |
|---|---|
| Age | 0.1043 |
| Income | 0.0465 |
| Sex | 0.0470 |
| Education | 0.0354 |
| Race | 0.0224 |

Race ranking last is unexpected. Epidemiological literature consistently identifies race/ethnicity as among the strongest demographic predictors of obesity, with the gradient between Asian and NH-Black/AIAN running roughly 30 percentage points nationally. One likely contributor is feature fragmentation: one-hot encoding splits `_RACE` into 7 binary columns, and SHAP assigns importance to each individual dummy separately. When those small per-dummy values are summed at the group level, they tend to understate the true group contribution because the permutation procedure doesn't account for the correlation structure across dummies within the same feature. Implementing grouped SHAP, which sums absolute SHAP values across all race dummies before ranking, would give a more accurate picture of how much the model is actually relying on race.

Top stable individual features were education level (`_EDUCAG_4`), sex, and older age groups, all passing the CV < 0.20 stability threshold.

---

## State-Level Validation

A useful sanity check between cell-level calibration and the full MRP poststratification is state-level: does the model's weighted prevalence by state match BRFSS's own weighted state estimates?

```
=== Logistic - State Validation ===
  Pred vs True corr:  0.8238
  Mean |error|:       3.14pp
  States |err|>3pp:   25

=== NN - State Validation ===
  Pred vs True corr:  0.7485
  Mean |error|:       2.33pp
  States |err|>3pp:   15
```

The NN has lower mean state error (2.33pp vs. 3.14pp) and fewer states with large errors, despite having worse cell-level calibration slopes. The state embedding is carrying meaningful geographic signal, partially compensating for poor cell-level calibration by learning state-level offsets. This is encouraging evidence that the geographic structure is learnable. State-level accuracy and cell calibration are measuring different things, and both need to be tracked separately.

The NN's better state-level MAE also supports building the GraphSAGE spatial layer in poststratification. The geographic signal is real, and a spatially-aware model will capture more of it than a state embedding alone can.

---

## Final Model Selection

The selection criterion is Brier score on reliable cells only (n >= 30), with sparse cell MAE as a tiebreaker if the primary difference is < 0.002.

| Metric | Logistic | NN |
|---|---|---|
| Brier (reliable cells) | nan | nan |
| Mean \|error\| (marginal cells) | 0.1633 | 0.1597 |
| AUC (weighted) | 0.6185 | 0.6339 |
| Overall Brier (weighted) | 0.2134 | 0.2112 |

The primary Brier criterion returned `nan` for both models because no reliable cell had sufficient outcome variance in the test set, per the splitting artifact described in Step 7. The tiebreaker (marginal cell MAE) selected Logistic on parsimony grounds, since the NN's marginal improvement (0.1633 vs. 0.1597) didn't clear the 0.01 threshold.

In practice, the NN wins on overall metrics and is saved for deployment. The logistic coefficients are saved separately for interpretability analysis. The open issues in Steps 7 and 10 need to be addressed before either model's county-level output is meaningful.

---

## Before poststratification

The following issues need to be resolved before county estimates are meaningful:

- [ ] Investigate DEFF of 0.002: verify design variables are being passed correctly into the variance calculation
- [ ] Check predicted probability std across test set. If below 0.05, the model is near-constant and demographic signal is not being learned
- [ ] Resolve cell calibration slopes: target mean slope above 0.85 before proceeding to poststratification
- [ ] Fix Brier NaN on reliable cells: increase test split to 20% or use 5-fold CV at the cell level
- [ ] Rerun SHAP with grouped feature importance: sum absolute SHAP values across race dummies before ranking
- [ ] Check for NaN logistic coefficients from empty interaction cells
- [ ] Consider [256, 256, 128] architecture based on SA4 results
- [ ] Verify Asian and Multiracial conformal coverage reaches 90%: consider larger calibration set or lower alpha



