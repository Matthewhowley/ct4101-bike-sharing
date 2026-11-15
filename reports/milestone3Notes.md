# Milestone 3: Model comparison

## Evaluation protocol

All models use the same time split. The training data is all hours before the 1st of October 2012. The test set is from the 1st of October to the 31st of December 2012 and was not used for model choice or tuning.

Features: `season`, `yr`, `mnth`, `hr`, `holiday`, `weekday`, `workingday`, `weathersit`, `temp`, `atemp`, `hum`, `windspeed`.  
Dropped: `instant` (index), `casual` and `registered`.  
`dteday` is used only to order the data and define the split.

Metrics: Mean Absolute Error, Root Mean Squared Error, R² on the held-out test set.

## Model settings

- **Hour of day mean:** predict the training mean `cnt` for each hour of day. No fitted parameters beyond those means.
- **Decision Tree:** `max_depth=10`, `min_samples_leaf=20`, `random_state=42`. Depth and leaf size were fixed in advance to limit overfitting.
- **Random Forest:** `n_estimators=100`, `max_depth=12`, `min_samples_leaf=10`, `random_state=42`. Settings were fixed in advance. No grid search and no use of the test set for tuning.

## Results

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Hour-of-day mean (baseline) | 94.91 | 142.38 | 0.501 |
| Decision Tree (max_depth=10) | 65.41 | 100.36 | 0.752 |
| Random Forest (n=100, max_depth=12) | 53.42 | 80.97 | 0.839 |

Random Forest is the best of the three on every metric. The Decision Tree improves on the baseline and the forest improves further on the tree.

## Residual and error analysis (Random Forest)

Mean absolute error by hour on the test set:

| Hour | MAE | Hour | MAE |
|---:|---:|---:|---:|
| 0 | 23.9 | 12 | 69.0 |
| 1 | 16.7 | 13 | 62.6 |
| 2 | 10.0 | 14 | 54.4 |
| 3 | 5.0 | 15 | 71.9 |
| 4 | 3.1 | 16 | 88.8 |
| 5 | 6.5 | 17 | 134.1 |
| 6 | 22.8 | 18 | 107.6 |
| 7 | 78.8 | 19 | 83.5 |
| 8 | 128.1 | 20 | 54.4 |
| 9 | 61.8 | 21 | 43.9 |
| 10 | 39.1 | 22 | 37.9 |
| 11 | 43.6 | 23 | 30.6 |

Mean absolute error by weather situation:

| weathersit | MAE |
|---:|---:|
| 1 (clear / partly cloudy) | 53.9 |
| 2 (mist / cloudy) | 49.6 |
| 3 (light rain or snow) | 68.4 |

Category 4 heavy rain or ice does not appear in this test window.

Error is lowest in the early hours with about 3–6 rentals around 03:00 - 05:00 and highest at the commuting times around hour 8 (MAE 128.1) and hour 17 (MAE 134.1), with hours 7, 16, 18 and 19 also with higher rates. Demand at those times is large and variable, so the model captures the average pattern but misses the extremes. Clear and misty hours have similar error but light rain or snow is worse. The residual plot (`reports/rf_residuals.png`) shows more scatter at high predicted counts, which matches the errors at peak hours. Hour, temperature and working-day related variables, are the most prominent in Feature importance (`reports/rf_feature_importance.png`) which is consistent with a commuting-driven rental pattern.

## Conclusion

Random Forest is the current best model on the chronological hold-out(MAE 53.42, RMSE 80.97, R² 0.839), ahead of the Decision Tree (MAE 65.41, R² 0.752) and the hour-of-day baseline (MAE 94.91, R² 0.501). I prefixed the settings and the test set was not used for tuning. 

Limitations: the test period is one year in autumn/winter in one city where peak hours and poorer weather remain hard to predict and rare severe weather is almost absent from the test block. In future i'll do light time blocked tuning inside the training period only, and a closer check of working-day versus weekend peaks.

## Files
Runnable comparison
- `src/model_comparison.py`

Results table
- `reports/model_comparison.csv`

Residual plot
- `reports/rf_residuals.png`

Feature importance
- `reports/rf_feature_importance.png` 