# Literature Comparison

**Note:** Direct comparison is approximate. Our dataset is synthetic (calibrated from PaySim), while literature results are on the original PaySim or real banking data. Feature sets and evaluation protocols differ.

| Study                        | Dataset                       | Method                    | Features                   | ROC-AUC   | PR-AUC    | Precision   | Recall   | Notes                                                  |
|:-----------------------------|:------------------------------|:--------------------------|:---------------------------|:----------|:----------|:------------|:---------|:-------------------------------------------------------|
| Lopez-Rojas et al. (2016)    | PaySim (6.3M txn)             | Random Forest             | Transactional only         | 0.97      | —         | —           | 0.95     | Original PaySim paper; full real-value features        |
| Carcillo et al. (2018)       | Real bank data                | Random Forest (streaming) | Transactional + aggregated | —         | 0.52      | —           | —        | Real-time fraud detection with concept drift           |
| Alarfaj et al. (2022)        | Credit card (Kaggle)          | XGBoost                   | PCA-transformed            | 0.98      | —         | 0.95        | 0.94     | Highly engineered, PCA features                        |
| Hilal et al. (2022)          | PaySim + others               | LightGBM                  | Transactional              | 0.99      | —         | 0.97        | 0.95     | Survey best-case; trained on original PaySim labels    |
| Kaggle community (2020-2023) | PaySim (Kaggle)               | XGBoost / LightGBM        | Transactional + engineered | 0.97–0.99 | 0.85–0.95 | —           | —        | Leaderboard solutions; risk of leakage in some entries |
| This work                    | Synthetic (PaySim-calibrated) | Dummy (Prior)             | Behavioral                 | 0.5       | 0.174     | —           | —        | Threshold=0.5 (prec≥0.5)                               |
| This work                    | Synthetic (PaySim-calibrated) | Dummy (Prior)             | Transactional              | 0.5       | 0.2695    | —           | —        | Threshold=0.5 (prec≥0.5)                               |
| This work                    | Synthetic (PaySim-calibrated) | Dummy (Prior)             | Combined                   | 0.5       | 0.2695    | —           | —        | Threshold=0.5 (prec≥0.5)                               |
| This work                    | Synthetic (PaySim-calibrated) | Logistic Regression       | Behavioral                 | 0.9013    | 0.836     | —           | —        | Threshold=0.35 (prec≥0.5)                              |
| This work                    | Synthetic (PaySim-calibrated) | Logistic Regression       | Transactional              | 0.5679    | 0.3342    | —           | —        | Threshold=0.6 (prec≥0.5)                               |
| This work                    | Synthetic (PaySim-calibrated) | Logistic Regression       | Combined                   | 0.5682    | 0.3342    | —           | —        | Threshold=0.6 (prec≥0.5)                               |
| This work                    | Synthetic (PaySim-calibrated) | Random Forest             | Behavioral                 | 0.9066    | 0.8404    | —           | —        | Threshold=0.15 (prec≥0.5)                              |
| This work                    | Synthetic (PaySim-calibrated) | Random Forest             | Transactional              | 0.7224    | 0.4391    | —           | —        | Threshold=0.65 (prec≥0.5)                              |
| This work                    | Synthetic (PaySim-calibrated) | Random Forest             | Combined                   | 0.7815    | 0.5681    | —           | —        | Threshold=0.5 (prec≥0.5)                               |
| This work                    | Synthetic (PaySim-calibrated) | Gradient Boosting         | Behavioral                 | 0.9005    | 0.8361    | —           | —        | Threshold=0.1 (prec≥0.5)                               |
| This work                    | Synthetic (PaySim-calibrated) | Gradient Boosting         | Transactional              | 0.7242    | 0.4498    | —           | —        | Threshold=0.45 (prec≥0.5)                              |
| This work                    | Synthetic (PaySim-calibrated) | Gradient Boosting         | Combined                   | 0.7806    | 0.5733    | —           | —        | Threshold=0.3 (prec≥0.5)                               |
