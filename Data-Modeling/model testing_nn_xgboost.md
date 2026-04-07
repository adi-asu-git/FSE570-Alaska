## Modeling: Neural Network and XGBoost

We evaluate two modeling approaches, a neural network and a gradient boosting model (XGBoost). These models are used to test how well different approaches capture variation in obesity risk across demographic groups, and to assess how model behavior might carry forward into the poststratification step.

A key consideration in this setup is how to handle survey weights. While weights help reflect the underlying population, they can also affect how much variation the model is able to learn across demographic groups.

---

### Neural Network

An embedding-based neural network was used to model obesity risk. Each categorical variable (age group, sex, race, education, income, and year) is represented using a learned embedding, rather than one-hot encoding. These embeddings are then concatenated and passed through a series of fully connected layers.

The model architecture consists of:
- embedding layers for each categorical feature  
- three fully connected layers with sizes:
  - 256 → 128 → 64  
- ReLU activation after each layer  
- batch normalization for stability  
- dropout = 0.10 for regularization  
- final linear layer producing a logit, followed by a sigmoid  

This setup allows the model to capture interactions between demographic variables without explicitly specifying them. For example, relationships between age, income, and race can be learned directly from the data.

To understand how survey weights affect training, three versions of the model were tested:
- fully weighted  
- softened weights (clipped + square root transformed)  
- unweighted  

These variations were used to evaluate how weighting influences subgroup-level predictions.

The architecture itself was kept consistent across all three runs, so that differences in performance could be attributed primarily to how weights were handled.

#### Results

| Model             | AUC   | Brier | Mean Cell Error |
|------------------|------|-------|----------------|
| Full weighted    | 0.6206 | 0.2190 | 0.2039 |
| Softened weights | 0.6216 | 0.2194 | 0.1658 |
| Unweighted       | 0.6183 | 0.2196 | **0.1469** |

At a high level, all three models perform similarly in terms of standard metrics. AUC remains around 0.62, and Brier scores are nearly identical. This suggests that, at the aggregate level, the model is capturing a consistent signal regardless of how weights are applied.

However, the differences become much more apparent when examining subgroup behavior. The fully weighted model produces the highest mean absolute error across demographic cells, indicating that predictions are overly compressed toward the overall average. This suggests that the model is not effectively distinguishing between different combinations of demographic characteristics.

Softening the weights improves this considerably. By reducing the influence of extreme weights, the model is able to recover more variation across groups while still incorporating some information about the sampling design. The unweighted model goes further, producing the lowest cell-level error and the strongest separation between demographic cells.

Looking at prediction variability provides additional insight. Across all models, predicted probabilities have a standard deviation of roughly 0.10, indicating that predictions are still relatively concentrated. However, the fully weighted model is the most compressed, while the unweighted model allows slightly more spread, which aligns with its improved cell-level performance.

Survey weights improve representativeness but can reduce the model’s ability to learn meaningful differences across demographic groups. In this case, the weighting scheme appears to smooth out variation that is important for the next stage of the analysis.

Because poststratification depends on subgroup-level predictions rather than individual classification accuracy, this loss of variation is a critical issue. A model that performs slightly better in terms of AUC but fails to distinguish between demographic cells may ultimately produce less reliable population estimates. The neural network results suggest that the underlying model is capable of capturing subgroup differences, but that these differences are sensitive to how weights are handled.

---

### XGBoost

A gradient boosting model (XGBoost) was trained using the same features and binning structure. XGBoost builds an ensemble of decision trees sequentially, where each new tree focuses on correcting errors from previous ones.

Several parameter configurations were tested, focusing on moderate tree depth, subsampling, and regularization to balance flexibility and stability. The final model was selected based on validation performance, using a combination of Brier score and cell-level calibration error.

The final model used:
- number of trees (n_estimators): 1200  
- learning rate: 0.025  
- max depth: 4  
- min_child_weight: 25  
- subsample: 0.8  
- colsample_bytree: 0.8  
- L1 regularization (reg_alpha): 0.2  
- L2 regularization (reg_lambda): 8.0  

Softened survey weights were used during training to reduce the influence of extreme observations while still incorporating some population structure.

#### Results

| Model             | AUC   | Brier | Mean Cell Error |
|------------------|------|-------|----------------|
| XGBoost (final)  | **0.6228** | **0.2192** | 0.1644 |

XGBoost achieves slightly better performance at the aggregate level, with marginal improvements in both AUC and Brier score compared to the neural network. This indicates that it is able to capture general patterns in the data somewhat more effectively.

However, these improvements do not translate as strongly to subgroup-level performance. The mean cell error remains relatively high, suggesting that predictions are still somewhat compressed across demographic groups. In other words, while the model is performing well overall, it is not fully capturing differences between specific demographic combinations.

Post-hoc calibration using isotonic regression was also applied, but this had minimal impact on performance. This suggests that the limitation is not primarily due to poorly calibrated probabilities, but rather that the model is not generating enough variation across cells to begin with.

Prediction spread remains similar to the neural network (standard deviation around 0.10), reinforcing the idea that both models are producing relatively conservative estimates. Even with a flexible model capable of capturing complex interactions, subgroup-level differentiation remains limited.

Compared to the neural network, XGBoost:
- provides slightly better overall performance  
- does not improve subgroup calibration  
- still produces relatively flat predictions across cells  

---

### Comparison

While the neural network and XGBoost differ slightly in their strengths, they converge to similar levels of overall performance. The key difference lies in how they handle subgroup variation:

- The neural network (especially the unweighted version) produces stronger differentiation across demographic cells  
- XGBoost provides slightly better overall fit but with less variation across groups  

Since the next step in the pipeline relies on subgroup-level predictions, this distinction is more important than small differences in AUC.
At this stage, both models are performing as expected, and the modeling pipeline is functioning correctly. However, the results suggest that the primary limitation lies in the available signal rather than the choice of model.

Demographic variables alone appear to explain only a limited portion of variation in obesity outcomes, and both models produce relatively conservative predictions as a result. The challenge moving forward is to preserve enough variation across demographic cells so that poststratification can produce meaningful differences at the county level.
