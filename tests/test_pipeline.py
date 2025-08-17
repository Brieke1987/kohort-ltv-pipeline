"""Test suite for LTV prediction pipeline."""

import unittest
from pathlib import Path
import tempfile
import shutil
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import sys
import os
import warnings

# Suppress warnings during tests
warnings.filterwarnings('ignore')

# Add src directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from data_ingestion import DataIngestion
from preprocessing import DataPreprocessor
from feature_engineering import FeatureEngineer
from modeling import LTVModeler
from monitoring import DriftDetector
from main import LTVPipeline


class TestDataIngestion(unittest.TestCase):
    """Test data ingestion component."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.ingestion = DataIngestion(
            max_missing_rate=0.1,
            min_records=10,
            max_duplicate_rate=0.05
        )
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def create_sample_data(self, n_rows=100):
        """Create sample cohort data for testing."""
        np.random.seed(42)
        
        data = {
            'day': pd.date_range('2024-01-01', periods=n_rows, freq='D').astype(str),
            'game_id': np.random.choice(['game_1', 'game_2'], n_rows),
            'game_genre': np.random.choice(['RPG', 'Strategy', 'Casual'], n_rows),
            'os_name': np.random.choice(['iOS', 'Android'], n_rows),
            'country_code': np.random.choice(['US', 'GB', 'DE'], n_rows),
            'channel': np.random.choice(['facebook', 'google', 'unity'], n_rows),
            'campaign_id': [f'camp_{i}' for i in range(n_rows)],
            'installs': np.random.randint(10, 1000, n_rows),
            'network_cost': np.random.uniform(100, 5000, n_rows),
            'retained_users_d1': np.random.randint(5, 500, n_rows),
            'retained_users_d3': np.random.randint(3, 400, n_rows),
            'retained_users_d7': np.random.randint(2, 300, n_rows),
            'retained_users_d14': np.random.randint(1, 200, n_rows),
            'retained_users_d30': np.random.randint(1, 100, n_rows),
            'revenue_d1': np.random.uniform(10, 1000, n_rows),
            'revenue_d3': np.random.uniform(20, 2000, n_rows),
            'revenue_d7': np.random.uniform(50, 3000, n_rows),
            'revenue_d14': np.random.uniform(100, 5000, n_rows),
            'revenue_d30': np.random.uniform(200, 8000, n_rows),
            'revenue_d60': np.random.uniform(300, 12000, n_rows),
            'revenue_d90': np.random.uniform(500, 15000, n_rows),
            'iap_revenue_d30': np.random.uniform(100, 4000, n_rows),
            'ad_revenue_d30': np.random.uniform(100, 4000, n_rows),
            'sessions_d1': np.random.uniform(1, 10, n_rows),
            'sessions_d7': np.random.uniform(5, 50, n_rows),
            'payers_d7': np.random.randint(0, 50, n_rows),
            'payers_d30': np.random.randint(0, 100, n_rows)
        }
        
        return pd.DataFrame(data)
    
    def test_load_data(self):
        """Test data loading functionality."""
        # Create test parquet file
        df = self.create_sample_data()
        test_file = Path(self.temp_dir) / "test_data.parquet"
        df.to_parquet(test_file)
        
        # Load data
        loaded_df = self.ingestion.load_data(test_file)
        
        self.assertEqual(len(loaded_df), 100)
        self.assertIsInstance(loaded_df, pd.DataFrame)
    
    def test_load_data_file_not_found(self):
        """Test error handling for missing file."""
        with self.assertRaises(FileNotFoundError):
            self.ingestion.load_data("non_existent_file.parquet")
    
    def test_validate_schema(self):
        """Test schema validation."""
        df = self.create_sample_data()
        
        # Should not raise error for valid schema
        self.ingestion.validate_schema(df)
        
        # Test with missing column
        df_missing = df.drop(columns=['installs'])
        with self.assertRaises(ValueError):
            self.ingestion.validate_schema(df_missing)
    
    def test_calculate_quality_metrics(self):
        """Test quality metrics calculation."""
        df = self.create_sample_data()
        
        quality_report = self.ingestion.calculate_quality_metrics(df)
        
        self.assertIn('total_records', quality_report)
        self.assertIn('duplicates', quality_report)
        self.assertIn('missing_rates', quality_report)
        self.assertEqual(quality_report['total_records'], 100)
    
    def test_validate_quality(self):
        """Test quality validation."""
        # Test with good quality data
        good_report = {
            'total_records': 1000,
            'duplicates': {'rate': 0.01},
            'missing_rates': {}
        }
        
        # Should not raise error
        self.ingestion.validate_quality(good_report)
        
        # Test with poor quality data
        bad_report = {
            'total_records': 5,  # Too few records
            'duplicates': {'rate': 0.1},  # High duplicate rate
            'missing_rates': {'critical_col': 0.2}  # High missing rate
        }
        
        with self.assertRaises(ValueError):
            self.ingestion.validate_quality(bad_report)


class TestDataPreprocessor(unittest.TestCase):
    """Test data preprocessing component."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.preprocessor = DataPreprocessor()
        
    def create_dirty_data(self):
        """Create data with quality issues for testing."""
        data = {
            'campaign_id': ['camp_1', None, 'camp_3'],
            'network_cost': [100, None, 200],
            'channel': ['Facebook', 'fb', 'GOOGLE'],
            'country_code': ['US', 'usa', 'GB'],
            'installs': [100, -10, 200],  # Negative value
            'retained_users_d1': [50, 150, 80],  # Middle one exceeds installs
            'revenue_d1': [100, 200, 300],
            'revenue_d7': [200, 150, 400],  # Middle one less than d1
            'revenue_d30': [500, 600, 700],
            'revenue_d90': [1000, 1200, 1400]
        }
        return pd.DataFrame(data)
    
    def test_handle_missing_values(self):
        """Test missing value handling."""
        df = self.create_dirty_data()
        
        df_clean = self.preprocessor.handle_missing_values(df)
        
        # Check campaign_id filled
        self.assertFalse(df_clean['campaign_id'].isnull().any())
        self.assertEqual(df_clean['campaign_id'].iloc[1], 'unknown')
        
        # Check network_cost filled
        self.assertFalse(df_clean['network_cost'].isnull().any())
    
    def test_standardize_text_fields(self):
        """Test text standardization."""
        df = self.create_dirty_data()
        
        df_clean = self.preprocessor.standardize_text_fields(df)
        
        # Check channel standardization
        self.assertIn('facebook', df_clean['channel'].values)
        self.assertIn('google', df_clean['channel'].values)
        
        # Check country code standardization
        self.assertIn('US', df_clean['country_code'].values)
        self.assertNotIn('usa', df_clean['country_code'].values)
    
    def test_fix_data_inconsistencies(self):
        """Test data inconsistency fixes."""
        df = self.create_dirty_data()
        
        df_clean = self.preprocessor.fix_data_inconsistencies(df)
        
        # Check negative values fixed
        self.assertTrue((df_clean['installs'] >= 0).all())
        
        # Check retention doesn't exceed installs
        self.assertTrue((df_clean['retained_users_d1'] <= df_clean['installs']).all())
        
        # Check revenue progression
        self.assertTrue((df_clean['revenue_d7'] >= df_clean['revenue_d1']).all())
    
    def test_cap_outliers(self):
        """Test outlier capping."""
        df = pd.DataFrame({
            'network_cost': [100, 200, 300, 10000],  # Last one is outlier
            'revenue_d30': [100, 200, 300, 50000]  # Last one is outlier
        })
        
        df_clean = self.preprocessor.cap_outliers(df)
        
        # Check that extreme values are capped
        self.assertTrue(df_clean['network_cost'].iloc[-1] < 10000)
        self.assertTrue(df_clean['revenue_d30'].iloc[-1] < 50000)


class TestFeatureEngineer(unittest.TestCase):
    """Test feature engineering component."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.engineer = FeatureEngineer()
        
    def create_clean_data(self):
        """Create clean data for feature engineering."""
        np.random.seed(42)
        n = 100
        
        data = {
            'installs': np.random.randint(100, 1000, n),
            'network_cost': np.random.uniform(500, 5000, n),
            'retained_users_d1': np.random.randint(50, 500, n),
            'retained_users_d7': np.random.randint(30, 300, n),
            'retained_users_d30': np.random.randint(10, 100, n),
            'revenue_d1': np.random.uniform(100, 1000, n),
            'revenue_d7': np.random.uniform(500, 3000, n),
            'revenue_d30': np.random.uniform(1000, 8000, n),
            'revenue_d90': np.random.uniform(2000, 15000, n),
            'payers_d7': np.random.randint(5, 50, n),
            'payers_d30': np.random.randint(10, 100, n),
            'sessions_d1': np.random.uniform(1, 10, n),
            'sessions_d7': np.random.uniform(10, 50, n),
            'iap_revenue_d30': np.random.uniform(500, 4000, n),
            'ad_revenue_d30': np.random.uniform(500, 4000, n),
            'game_genre': np.random.choice(['RPG', 'Strategy'], n),
            'os_name': np.random.choice(['iOS', 'Android'], n),
            'country_code': np.random.choice(['US', 'GB'], n),
            'channel': np.random.choice(['facebook', 'google'], n)
        }
        
        return pd.DataFrame(data)
    
    def test_create_retention_features(self):
        """Test retention feature creation."""
        df = self.create_clean_data()
        
        df_features = self.engineer.create_retention_features(df)
        
        # Check that features were added to the tracking list
        retention_features = [f for f in self.engineer.features_created if 'retention' in f]
        self.assertGreater(len(retention_features), 0)
        
        # Check specific features exist
        for feat in ['retention_rate_d1', 'retention_rate_d7', 'retention_rate_d30']:
            self.assertIn(feat, df_features.columns)
            # Verify rates are between 0 and 1
            self.assertTrue((df_features[feat] >= 0).all())
            self.assertTrue((df_features[feat] <= 1).all())
        
        # Check retention decay features
        if 'retention_decay_d1_d7' in df_features.columns:
            self.assertTrue((df_features['retention_decay_d1_d7'] >= 0).all())
    
    def test_create_revenue_features(self):
        """Test revenue feature creation."""
        df = self.create_clean_data()
        
        df_features = self.engineer.create_revenue_features(df)
        
        # Check ARPI features
        self.assertIn('arpi_d1', df_features.columns)
        self.assertIn('arpi_d7', df_features.columns)
        self.assertIn('arpi_d30', df_features.columns)
        
        # Check revenue growth features
        self.assertIn('revenue_growth_d1_d7', df_features.columns)
        
        # Verify ARPI is positive
        self.assertTrue((df_features['arpi_d1'] >= 0).all())
    
    def test_create_monetization_features(self):
        """Test monetization feature creation."""
        df = self.create_clean_data()
        
        df_features = self.engineer.create_monetization_features(df)
        
        # Check payer conversion features
        self.assertIn('payer_conversion_d7', df_features.columns)
        self.assertIn('payer_conversion_d30', df_features.columns)
        
        # Check ARPPU features
        self.assertIn('arppu_d7', df_features.columns)
        
        # Check IAP/Ad split features
        self.assertIn('iap_revenue_share', df_features.columns)
        self.assertIn('ad_revenue_share', df_features.columns)
        
        # Verify shares sum to ~1
        total_share = df_features['iap_revenue_share'] + df_features['ad_revenue_share']
        np.testing.assert_array_almost_equal(total_share, np.ones(len(df_features)), decimal=5)
    
    def test_create_engagement_features(self):
        """Test engagement feature creation."""
        df = self.create_clean_data()
        
        df_features = self.engineer.create_engagement_features(df)
        
        # Check features were created
        engagement_features = [f for f in self.engineer.features_created if 'session' in f or 'engagement' in f]
        self.assertGreater(len(engagement_features), 0)
        
        # Check specific features
        if 'sessions_per_install_d1' in df_features.columns:
            self.assertTrue((df_features['sessions_per_install_d1'] >= 0).all())
    
    def test_create_efficiency_features(self):
        """Test efficiency feature creation."""
        df = self.create_clean_data()
        
        df_features = self.engineer.create_efficiency_features(df)
        
        # Check CPI feature
        self.assertIn('cpi', df_features.columns)
        self.assertTrue((df_features['cpi'] >= 0).all())
        
        # Check ROAS features
        for day in [1, 7, 14, 30]:
            feat = f'roas_d{day}'
            if feat in df_features.columns:
                self.assertTrue((df_features[feat] >= 0).all())
    
    def test_create_categorical_encodings(self):
        """Test categorical encoding creation."""
        df = self.create_clean_data()
        
        df_features = self.engineer.create_categorical_encodings(df)
        
        # Check frequency encodings
        for col in ['game_genre', 'os_name', 'country_code', 'channel']:
            freq_col = f'{col}_frequency'
            if freq_col in df_features.columns:
                self.assertTrue((df_features[freq_col] > 0).all())
    
    def test_create_composite_scores(self):
        """Test composite score creation."""
        df = self.create_clean_data()
        
        # Create prerequisite features
        df = self.engineer.create_retention_features(df)
        df = self.engineer.create_revenue_features(df)
        df = self.engineer.create_engagement_features(df)
        
        # Create composite scores
        df_features = self.engineer.create_composite_scores(df)
        
        # Check scores exist and are in valid range
        if 'early_performance_score' in df_features.columns:
            self.assertTrue((df_features['early_performance_score'] >= 0).all())
            self.assertTrue((df_features['early_performance_score'] <= 1.5).all())  # Allow some overflow
        
        if 'cohort_quality_score' in df_features.columns:
            self.assertTrue((df_features['cohort_quality_score'] >= 0).all())
    
    def test_full_feature_pipeline(self):
        """Test complete feature engineering pipeline."""
        df = self.create_clean_data()
        
        # Run full pipeline
        df_final, feature_cols, stats = self.engineer.run(df)
        
        # Check output structure
        self.assertIsInstance(df_final, pd.DataFrame)
        self.assertIsInstance(feature_cols, list)
        self.assertIsInstance(stats, dict)
        
        # Check features were created
        self.assertGreater(len(feature_cols), 20)  # Should create many features
        
        # Check no nulls in features
        for col in feature_cols:
            self.assertFalse(df_final[col].isnull().any(), f"Column {col} has null values")


class TestLTVModeler(unittest.TestCase):
    """Test modeling component."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.modeler = LTVModeler(
            test_size=0.2,
            cv_folds=3,
            random_state=42
        )
        
    def create_modeling_data(self):
        """Create data for modeling tests."""
        np.random.seed(42)
        n = 500
        
        # Create features with some correlation to target
        X = pd.DataFrame({
            'feature_1': np.random.randn(n),
            'feature_2': np.random.randn(n),
            'feature_3': np.random.randn(n),
            'categorical_feat': np.random.choice(['A', 'B', 'C'], n)
        })
        
        # Create target with some dependency on features
        y = (
            X['feature_1'] * 100 + 
            X['feature_2'] * 50 + 
            np.random.randn(n) * 10 + 
            1000
        )
        
        # Add target column to dataframe
        df = X.copy()
        df['revenue_d90'] = y
        
        return df, list(X.columns)
    
    def test_prepare_data(self):
        """Test data preparation."""
        df, feature_cols = self.create_modeling_data()
        
        X, y = self.modeler.prepare_data(df, feature_cols)
        
        # Check shapes
        self.assertEqual(len(X), len(y))
        self.assertEqual(list(X.columns), feature_cols)
        
        # Check target is non-negative
        self.assertTrue((y >= 0).all())
    
    def test_encode_categoricals(self):
        """Test categorical encoding."""
        df, _ = self.create_modeling_data()
        X = df[['categorical_feat', 'feature_1']]
        
        X_encoded = self.modeler.encode_categoricals(X, fit=True)
        
        # Check categorical column is encoded
        self.assertTrue(pd.api.types.is_numeric_dtype(X_encoded['categorical_feat']))
        
        # Check encoder is stored
        self.assertIn('categorical_feat', self.modeler.label_encoders)
    
    def test_train_model(self):
        """Test model training."""
        df, feature_cols = self.create_modeling_data()
        X, y = self.modeler.prepare_data(df, feature_cols)
        
        # Encode categoricals
        X = self.modeler.encode_categoricals(X, fit=True)
        
        # Split data
        X_train = X[:400]
        y_train = y[:400]
        X_val = X[400:]
        y_val = y[400:]
        
        model = self.modeler.train(X_train, y_train, X_val, y_val)
        
        # Check model is trained
        self.assertIsNotNone(model)
        
        # Check predictions work
        predictions = model.predict(X_val)
        self.assertEqual(len(predictions), len(X_val))
    
    def test_cross_validate(self):
        """Test cross-validation."""
        df, feature_cols = self.create_modeling_data()
        X, y = self.modeler.prepare_data(df, feature_cols)
        X = self.modeler.encode_categoricals(X, fit=True)
        
        # Run cross-validation
        cv_results = self.modeler.cross_validate(X, y)
        
        # Check results structure
        self.assertIn('rmse_mean', cv_results)
        self.assertIn('rmse_std', cv_results)
        self.assertIn('r2_mean', cv_results)
        self.assertIn('r2_std', cv_results)
        
        # Check values are reasonable
        self.assertGreater(cv_results['rmse_mean'], 0)
        self.assertGreaterEqual(cv_results['r2_mean'], -1)  # R2 can be negative for bad models
        self.assertLessEqual(cv_results['r2_mean'], 1)
    
    def test_extract_feature_importance(self):
        """Test feature importance extraction."""
        df, feature_cols = self.create_modeling_data()
        X, y = self.modeler.prepare_data(df, feature_cols)
        X = self.modeler.encode_categoricals(X, fit=True)
        
        # Train model
        model = self.modeler.train(X, y)
        
        # Extract importance
        importance = self.modeler.extract_feature_importance(model)
        
        # Check structure
        self.assertIsInstance(importance, dict)
        self.assertEqual(len(importance), len(self.modeler.feature_names))
        
        # Check values
        for feat, imp in importance.items():
            self.assertGreaterEqual(imp, 0)
            self.assertLessEqual(imp, 1)
    
    def test_save_model(self):
        """Test model saving."""
        df, feature_cols = self.create_modeling_data()
        X, y = self.modeler.prepare_data(df, feature_cols)
        X = self.modeler.encode_categoricals(X, fit=True)
        
        # Train model
        model = self.modeler.train(X, y)
        self.modeler.feature_importance = {'feat1': 0.5}
        self.modeler.metrics = {'test': {'r2': 0.8}}
        
        # Save model
        save_dir = Path(self.temp_dir) / "test_model"
        self.modeler.save_model(model, save_dir)
        
        # Check files exist
        self.assertTrue((save_dir / "model.joblib").exists())
        self.assertTrue((save_dir / "model_metadata.json").exists())
        
        # Load and verify metadata
        import json
        with open(save_dir / "model_metadata.json") as f:
            metadata = json.load(f)
        
        self.assertIn('target_column', metadata)
        self.assertIn('feature_importance', metadata)
        self.assertIn('metrics', metadata)
    
    def test_full_modeling_pipeline(self):
        """Test complete modeling pipeline."""
        df, feature_cols = self.create_modeling_data()
        
        # Run full pipeline
        save_dir = Path(self.temp_dir) / "model_output"
        model, metrics = self.modeler.run(df, feature_cols, output_dir=save_dir)
        
        # Check outputs
        self.assertIsNotNone(model)
        self.assertIn('train', metrics)
        self.assertIn('test', metrics)
        self.assertIn('cv', metrics)
        self.assertIn('data', metrics)
        
        # Check model files saved
        self.assertTrue((save_dir / "model.joblib").exists())
        
        # Verify model can make predictions
        X = df[feature_cols]
        X_encoded = self.modeler.encode_categoricals(X, fit=False)
        predictions = model.predict(X_encoded)
        self.assertEqual(len(predictions), len(df))


class TestDriftDetector(unittest.TestCase):
    """Test drift detection component."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.detector = DriftDetector(n_bins=5)
        
    def create_baseline_data(self):
        """Create baseline data for drift detection."""
        np.random.seed(42)
        n = 1000
        
        data = pd.DataFrame({
            'feature_1': np.random.normal(0, 1, n),
            'feature_2': np.random.normal(10, 2, n),
            'categorical_feat': np.random.choice(['A', 'B', 'C'], n, p=[0.5, 0.3, 0.2]),
            'revenue_d90': np.random.uniform(1000, 5000, n)
        })
        
        return data
    
    def create_drifted_data(self):
        """Create data with drift."""
        np.random.seed(43)
        n = 1000
        
        data = pd.DataFrame({
            'feature_1': np.random.normal(0.5, 1.5, n),  # Mean and std shifted
            'feature_2': np.random.normal(12, 2, n),  # Mean shifted
            'categorical_feat': np.random.choice(['A', 'B', 'C', 'D'], n, p=[0.3, 0.3, 0.2, 0.2]),  # New category
            'revenue_d90': np.random.uniform(2000, 6000, n)  # Distribution shifted
        })
        
        return data
    
    def test_fit_baseline(self):
        """Test baseline fitting."""
        df = self.create_baseline_data()
        feature_cols = ['feature_1', 'feature_2', 'categorical_feat']
        
        baseline_stats = self.detector.fit_baseline(df, feature_cols, 'revenue_d90')
        
        # Check baseline statistics stored
        self.assertIsNotNone(self.detector.baseline_stats)
        self.assertIn('features', baseline_stats)
        self.assertIn('target', baseline_stats)
        
        # Check feature statistics
        self.assertIn('feature_1', baseline_stats['features'])
        self.assertIn('mean', baseline_stats['features']['feature_1'])
    
    def test_calculate_psi(self):
        """Test PSI calculation."""
        expected = np.array([0.2, 0.3, 0.3, 0.2])
        actual = np.array([0.1, 0.4, 0.3, 0.2])
        
        psi = self.detector.calculate_psi(expected, actual)
        
        # PSI should be positive
        self.assertGreater(psi, 0)
        
        # Test identical distributions
        psi_same = self.detector.calculate_psi(expected, expected)
        self.assertAlmostEqual(psi_same, 0, places=5)
    
    def test_detect_drift(self):
        """Test drift detection."""
        # Fit baseline
        baseline_df = self.create_baseline_data()
        feature_cols = ['feature_1', 'feature_2', 'categorical_feat']
        self.detector.fit_baseline(baseline_df, feature_cols, 'revenue_d90')
        
        # Test on drifted data
        drifted_df = self.create_drifted_data()
        drift_results = self.detector.detect_drift(drifted_df, feature_cols, 'revenue_d90')
        
        # Check results structure
        self.assertIn('features', drift_results)
        self.assertIn('summary', drift_results)
        
        # Should detect drift in feature_1
        self.assertIn('feature_1', drift_results['features'])
        self.assertTrue(drift_results['features']['feature_1']['drift_detected'])
        
        # Should detect new category in categorical_feat
        self.assertIn('categorical_feat', drift_results['features'])
        cat_drift = drift_results['features']['categorical_feat']
        self.assertIn('new_categories', cat_drift)
        self.assertIn('D', cat_drift['new_categories'])
    
    def test_get_alerts(self):
        """Test alert generation."""
        drift_results = {
            'features': {
                'feature_1': {'severity': 'high', 'drift_detected': True},
                'feature_2': {'severity': 'low', 'drift_detected': True}
            },
            'summary': {
                'drift_rate': 0.5,
                'n_features_checked': 2,
                'n_drifted_features': 2
            },
            'target': {
                'drift_detected': True,
                'psi': 0.3
            }
        }
        
        alerts = self.detector.get_alerts(drift_results)
        
        # Should generate alerts
        self.assertGreater(len(alerts), 0)
        
        # Should have high severity alert for feature drift
        high_alerts = [a for a in alerts if a['severity'] == 'HIGH']
        self.assertGreater(len(high_alerts), 0)


class TestLTVPipeline(unittest.TestCase):
    """Test main pipeline orchestration."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.pipeline = LTVPipeline(output_dir=self.temp_dir)
        
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def create_test_data_file(self):
        """Create test data file."""
        np.random.seed(42)
        n = 100
        
        data = {
            'day': pd.date_range('2024-01-01', periods=n, freq='D').astype(str),
            'game_id': [f'game_{i%3}' for i in range(n)],
            'game_genre': np.random.choice(['RPG', 'Strategy'], n),
            'os_name': np.random.choice(['iOS', 'Android'], n),
            'country_code': np.random.choice(['US', 'GB'], n),
            'channel': np.random.choice(['facebook', 'google'], n),
            'campaign_id': [f'camp_{i}' for i in range(n)],
            'installs': np.random.randint(100, 1000, n),
            'network_cost': np.random.uniform(500, 5000, n),
            'retained_users_d1': np.random.randint(50, 500, n),
            'retained_users_d3': np.random.randint(40, 400, n),
            'retained_users_d7': np.random.randint(30, 300, n),
            'retained_users_d14': np.random.randint(20, 200, n),
            'retained_users_d30': np.random.randint(10, 100, n),
            'revenue_d1': np.random.uniform(100, 1000, n),
            'revenue_d3': np.random.uniform(200, 2000, n),
            'revenue_d7': np.random.uniform(500, 3000, n),
            'revenue_d14': np.random.uniform(1000, 5000, n),
            'revenue_d30': np.random.uniform(2000, 8000, n),
            'revenue_d60': np.random.uniform(3000, 12000, n),
            'revenue_d90': np.random.uniform(5000, 15000, n),
            'iap_revenue_d30': np.random.uniform(1000, 4000, n),
            'ad_revenue_d30': np.random.uniform(1000, 4000, n),
            'sessions_d1': np.random.uniform(1, 10, n),
            'sessions_d7': np.random.uniform(10, 50, n),
            'payers_d7': np.random.randint(5, 50, n),
            'payers_d30': np.random.randint(10, 100, n)
        }
        
        df = pd.DataFrame(data)
        test_file = Path(self.temp_dir) / "test_cohort_data.parquet"
        df.to_parquet(test_file)
        
        return test_file
    
    def test_pipeline_initialization(self):
        """Test pipeline initialization."""
        self.assertIsNotNone(self.pipeline.components)
        self.assertIn('ingestion', self.pipeline.components)
        self.assertIn('preprocessor', self.pipeline.components)
        self.assertIn('feature_engineer', self.pipeline.components)
        self.assertIn('modeler', self.pipeline.components)
    
    def test_run_ingestion(self):
        """Test ingestion step."""
        test_file = self.create_test_data_file()
        
        success = self.pipeline.run_ingestion(test_file)
        
        self.assertTrue(success)
        self.assertIn('raw', self.pipeline.data)
        self.assertEqual(len(self.pipeline.data['raw']), 100)
    
    def test_run_preprocessing(self):
        """Test preprocessing step."""
        test_file = self.create_test_data_file()
        
        # First run ingestion
        self.pipeline.run_ingestion(test_file)
        
        # Then run preprocessing
        success = self.pipeline.run_preprocessing()
        
        self.assertTrue(success)
        self.assertIn('processed', self.pipeline.data)
        
    def test_run_feature_engineering(self):
        """Test feature engineering step."""
        test_file = self.create_test_data_file()
        
        # Run prerequisite steps
        self.pipeline.run_ingestion(test_file)
        self.pipeline.run_preprocessing()
        
        # Run feature engineering
        success = self.pipeline.run_feature_engineering()
        
        self.assertTrue(success)
        self.assertIn('features', self.pipeline.data)
        self.assertIn('feature_columns', self.pipeline.data)
    
    def test_run_modeling(self):
        """Test modeling step."""
        test_file = self.create_test_data_file()
        
        # Run prerequisite steps
        self.pipeline.run_ingestion(test_file)
        self.pipeline.run_preprocessing()
        self.pipeline.run_feature_engineering()
        
        # Run modeling
        success = self.pipeline.run_modeling()
        
        self.assertTrue(success)
        self.assertIn('ltv', self.pipeline.models)
        self.assertIn('modeling', self.pipeline.reports)
    
    def test_run_monitoring_and_export(self):
        """Test monitoring and export step."""
        test_file = self.create_test_data_file()
        
        # Run all prerequisite steps
        self.pipeline.run_ingestion(test_file)
        self.pipeline.run_preprocessing()
        self.pipeline.run_feature_engineering()
        self.pipeline.run_modeling()
        
        # Run monitoring and export
        success = self.pipeline.run_monitoring_and_export()
        
        self.assertTrue(success)
        self.assertIn('baseline', self.pipeline.reports)
        
        # Check export files were created
        export_dir = Path(self.temp_dir) / "exports"
        self.assertTrue(export_dir.exists())
        self.assertTrue((export_dir / "train_data.parquet").exists())
        self.assertTrue((export_dir / "test_data.parquet").exists())
        self.assertTrue((export_dir / "model_predictions.parquet").exists())
    
    def test_preprocessing_impact_display(self):
        """Test preprocessing impact summary."""
        test_file = self.create_test_data_file()
        
        # Run ingestion and preprocessing
        self.pipeline.run_ingestion(test_file)
        
        # Mock the _show_preprocessing_impact method to verify it's called
        with patch.object(self.pipeline, '_show_preprocessing_impact') as mock_show:
            self.pipeline.run_preprocessing()
            mock_show.assert_called_once()
    
    def test_export_datasets(self):
        """Test dataset export functionality."""
        test_file = self.create_test_data_file()
        
        # Run full pipeline up to modeling
        self.pipeline.run_ingestion(test_file)
        self.pipeline.run_preprocessing()
        self.pipeline.run_feature_engineering()
        self.pipeline.run_modeling()
        
        # Test export
        self.pipeline._export_datasets()
        
        # Verify exports
        export_dir = Path(self.temp_dir) / "exports"
        
        # Load and verify train data
        train_df = pd.read_parquet(export_dir / "train_data.parquet")
        self.assertIn('prediction', train_df.columns)
        self.assertIn('revenue_d90', train_df.columns)
        
        # Load and verify test data
        test_df = pd.read_parquet(export_dir / "test_data.parquet")
        self.assertIn('prediction', test_df.columns)
        
        # Verify split proportions
        total_records = len(train_df) + len(test_df)
        test_proportion = len(test_df) / total_records
        self.assertAlmostEqual(test_proportion, 0.2, places=1)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_empty_dataframe(self):
        """Test handling of empty dataframe."""
        ingestion = DataIngestion()
        empty_df = pd.DataFrame()
        
        with self.assertRaises(ValueError):
            ingestion.validate_schema(empty_df)
    
    def test_all_missing_values(self):
        """Test handling of all missing values."""
        preprocessor = DataPreprocessor()
        df = pd.DataFrame({
            'col1': [None, None, None],
            'col2': [np.nan, np.nan, np.nan]
        })
        
        df_clean = preprocessor.handle_missing_values(df)
        # Should fill with defaults
        self.assertFalse(df_clean.isnull().all().all())
    
    def test_single_unique_value(self):
        """Test handling of features with single unique value."""
        engineer = FeatureEngineer()
        df = pd.DataFrame({
            'installs': [100, 100, 100],
            'retained_users_d1': [50, 50, 50]
        })
        
        df_features = engineer.create_retention_features(df)
        # Should handle division without error
        self.assertIn('retention_rate_d1', df_features.columns)
    
    def test_extreme_outliers(self):
        """Test handling of extreme outliers."""
        preprocessor = DataPreprocessor(cap_outliers_percentile=0.95)
        df = pd.DataFrame({
            'network_cost': [100, 200, 300, 1000000]  # Extreme outlier
        })
        
        df_clean = preprocessor.cap_outliers(df)
        # Extreme value should be capped
        self.assertLess(df_clean['network_cost'].max(), 1000000)
    
    def test_missing_target_column(self):
        """Test handling of missing target column."""
        modeler = LTVModeler()
        df = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [4, 5, 6]
        })
        
        with self.assertRaises(ValueError):
            modeler.prepare_data(df, ['feature1', 'feature2'])
    
    def test_invalid_file_path(self):
        """Test handling of invalid file paths."""
        pipeline = LTVPipeline()
        
        success = pipeline.run_ingestion(Path("non_existent_file.parquet"))
        self.assertFalse(success)
    
    def test_pipeline_step_dependencies(self):
        """Test pipeline step dependency checking."""
        pipeline = LTVPipeline()
        
        # Try to run preprocessing without ingestion
        success = pipeline.run_preprocessing()
        self.assertFalse(success)
        
        # Try to run feature engineering without preprocessing
        success = pipeline.run_feature_engineering()
        self.assertFalse(success)
        
        # Try to run modeling without features
        success = pipeline.run_modeling()
        self.assertFalse(success)


if __name__ == '__main__':
    # Set up test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestDataIngestion))
    suite.addTests(loader.loadTestsFromTestCase(TestDataPreprocessor))
    suite.addTests(loader.loadTestsFromTestCase(TestFeatureEngineer))
    suite.addTests(loader.loadTestsFromTestCase(TestLTVModeler))
    suite.addTests(loader.loadTestsFromTestCase(TestDriftDetector))
    suite.addTests(loader.loadTestsFromTestCase(TestLTVPipeline))
    suite.addTests(loader.loadTestsFromTestCase(TestEdgeCases))
    
    # Run tests with verbosity
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1)