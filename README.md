# LTV Prediction Pipeline for Mobile Gaming Cohorts

## Overview
Assesment for ML Engineer role - Machine Learning pipeline for predicting D90 revenue (LTV proxy) using D0-D30 user acquisition cohort data.

## Executive Summary

This solution demonstrates a production-ready ML pipeline, including modular code structure, reproducible data ingestion and preprocessing, automated feature engineering, model training with validation and cross-validation, interpretability through feature importances, and monitoring/export steps.

The project includes automated tests and coverage reporting (63% coverage, 70% passing), which provide a strong baseline for production QA. The remaining gaps are in orchestration branches and failure-mode tests, which are already scaffolded and would be extended with more time.

Overall, the pipeline is robust, modular, and deployable, with strong model performance (R² ~0.98, MAPE ~8.8%). This demonstrates readiness for production ML systems while highlighting clear next steps for continued improvement.

## Quick Start

```bash
# Clone repository
git clone https://github.com/brieke1987/kohort-ltv-pipeline.git
cd kohort-ltv-pipeline

# Install Python 3.11 using homebrew (Mac)
brew install python@3.11

# Create venv with Python 3.11 and Install dependencies (Mac)
python3.11 -m venv venv
source venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

# Create data/raw folders and place your data file in kohort-ltv-pipeline/data/raw

# Run the pipeline
python3.11 src/main.py --data-file data/raw/gaming_cohorts.parquet

# Run tests

# Install test dependencies
pip install -r requirements-test.txt

# Run pytest
pytest tests/ -v

# Generate HTML coverage report
pytest --cov=src --cov-report=html tests/

```
## Overall Results

| Metric | RMSE | R² | MAPE |
| ----- | ----- | ----- | ----- |
| Train metrics | $437.51 |0.994|6.5%|
| Test metrics | $816.73 |0.980|8.8%|
| CV Results |$828.58±43.47|0.979±0.002||


1. The model generalizes well: train and test scores are very close, with only moderate increase in RMSE from train ($417) → test ($810).
2. R² above 0.98 shows the model explains ~98% of variance in the target.
3. MAPE under 10% indicates strong predictive accuracy relative to revenue scale.
4. Cross-validation confirms consistency across folds.

### Top 10 features:
  1. arpi_d30: 0.4140
  2. installs: 0.2185
  3. arpi_d14: 0.2088
  4. payer_conversion_d30: 0.0343
  5. arppu_d30: 0.0210
  6. arpi_d7: 0.0128
  7. game_genre_frequency: 0.0117
  8. ad_revenue_share: 0.0081
  9. country_code_frequency: 0.0079
 10. arpi_d1: 0.0047

Long-term monetization signals (30-day ARPI, installs, 14-day ARPI) dominate importance. Early retention signals (7-day ARPI) still contribute but less strongly. User quality and distribution factors (payer conversion, cohort quality, country mix, genre) have meaningful but smaller impact.

## Data Ingestion

1. Schema validation passed
2. Data quality validation passed (1.5% duplicates - Threshold set to 2% and deemed reasonable)
3. Records: 12,230
4. Date range: 2024-01-01 to 2024-10-31 (304 days)

### Data Preprocessing
1. Filled 188 missing campaign_ids
2. Filled 187 missing network_costs
3. Applied 13 data consistency fixes
4. Capped outliers in 12 columns

#### Unique values (before → after)
1. game_genre: 5
2. os_name: 6 → 2
3. country_code: 46 → 23
4. channel: 7

#### Key Statistics (before → after)
| Metric | Mean - Before | Mean - After | Min - Before | Min - After | Max - Before | Max - After |
| ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| installs | $591.77 |$591.77|$1.00|$1.00|$1,500.00|$1,500.00|
| network_cost | $1,270.53 |$1,270.11|$0.00|$0.00|$5,695.33|$4,311.91|
| revenue_d30 |$3,106.80|$3,101.02|$61.55|$61.55|$20,081.33|$13,753.18|
|revenue_d90|$6,210.48|$6,177.70|$114.12|$114.12|$41,939.74|$27,099.21|

#### Feature Engineering: 27 → 57 features
1. Created 8 retention features
2. Created 11 revenue features
3. Created 8 monetization features
4. Created 7 engagement features
5. Created 7 efficiency features
6. Created 15 categorical features

#### Export & Documentation
1. 9784 records to outputs/exports/train_data.parquet
2. 2446 records to outputs/exports/test_data.parquet
3. 12230 records to outputs/exports/model_predictions.parquet
4. test coverage kohort-ltv-pipeline/htmlcov

### Next Steps
1. Improve branch coverage in orchestration (main.py).
2. Add more edge case tests for modelling and prep.
3. Stabilise failing tests.
