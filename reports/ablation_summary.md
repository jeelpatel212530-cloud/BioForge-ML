# Ablation summary (scaffold splits, mean ± std over seeds)


## Test set

| model | RMSE | MAE | R² | Spearman |
|---|---|---|---|---|
| rf_ecfp4 | 0.837 ± 0.129 | 0.650 ± 0.109 | 0.603 ± 0.127 | 0.761 ± 0.079 |
| rf_ecfp4_desc_2d_3d | 0.852 ± 0.128 | 0.662 ± 0.108 | 0.588 ± 0.129 | 0.756 ± 0.078 |
| gbr_desc_2d_3d | 1.171 ± 0.071 | 0.942 ± 0.061 | 0.231 ± 0.122 | 0.503 ± 0.117 |
| gbr_desc_2d | 1.174 ± 0.082 | 0.930 ± 0.070 | 0.225 ± 0.138 | 0.505 ± 0.116 |
| baseline_mean | 1.386 ± 0.101 | 1.183 ± 0.098 | -0.065 ± 0.079 | 0.000 ± 0.000 |

## Val set

| model | RMSE | MAE | R² | Spearman |
|---|---|---|---|---|
| rf_ecfp4 | 0.788 ± 0.084 | 0.601 ± 0.058 | 0.647 ± 0.072 | 0.794 ± 0.041 |
| rf_ecfp4_desc_2d_3d | 0.800 ± 0.086 | 0.611 ± 0.053 | 0.637 ± 0.071 | 0.787 ± 0.038 |
| gbr_desc_2d | 1.099 ± 0.069 | 0.876 ± 0.050 | 0.317 ± 0.065 | 0.573 ± 0.044 |
| gbr_desc_2d_3d | 1.102 ± 0.048 | 0.874 ± 0.036 | 0.313 ± 0.049 | 0.575 ± 0.039 |
| baseline_mean | 1.349 ± 0.048 | 1.162 ± 0.037 | -0.028 ± 0.024 | 0.000 ± 0.000 |

**Effect of 3D features** (GBR, test RMSE, `desc_2d_3d` minus `desc_2d`, per seed): mean -0.003 ± 0.018 pIC50 units (negative = 3D helps; helps in 3/5 splits).

**Sanity check vs. predicting the mean:** `rf_ecfp4` test RMSE 0.837 vs baseline 1.386 -> beats the baseline.
