"""Feature engineering module for LTV prediction pipeline."""

import logging
from typing import Dict, Any, Tuple, List, Optional

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Create features for LTV prediction from cohort data."""
    
    # Core metrics that should exist in the data
    RETENTION_DAYS = [1, 3, 7, 14, 30]
    REVENUE_DAYS = [1, 3, 7, 14, 30, 60, 90]
    PAYER_DAYS = [7, 30]
    SESSION_DAYS = [1, 7]
    
    def __init__(self):
        """Initialize feature engineer."""
        self.features_created = []
        self.feature_importance = {}
    
    def create_retention_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create retention-based features.
        
        Args:
            df: Input DataFrame with cohort data
            
        Returns:
            DataFrame with retention features added
        """
        # Retention rates
        for day in self.RETENTION_DAYS:
            col = f'retained_users_d{day}'
            if col in df.columns:
                feature_name = f'retention_rate_d{day}'
                df[feature_name] = df[col] / df['installs'].clip(lower=1)
                self.features_created.append(feature_name)
        
        # Retention decay (how fast users churn)
        if all(f'retention_rate_d{d}' in df.columns for d in [1, 7]):
            df['retention_decay_d1_d7'] = (
                df['retention_rate_d1'] - df['retention_rate_d7']
            ).clip(lower=0)
            self.features_created.append('retention_decay_d1_d7')
        
        if all(f'retention_rate_d{d}' in df.columns for d in [7, 30]):
            df['retention_decay_d7_d30'] = (
                df['retention_rate_d7'] - df['retention_rate_d30']
            ).clip(lower=0)
            self.features_created.append('retention_decay_d7_d30')
        
        # Retention stability (long-term vs short-term)
        if all(f'retention_rate_d{d}' in df.columns for d in [1, 30]):
            df['retention_stability'] = (
                df['retention_rate_d30'] / df['retention_rate_d1'].clip(lower=0.001)
            )
            self.features_created.append('retention_stability')
        
        logger.info(f"Created {len([f for f in self.features_created if 'retention' in f])} retention features")
        return df
    
    def create_revenue_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create revenue-based features.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with revenue features added
        """
        # ARPI (Average Revenue Per Install)
        for day in [1, 7, 14, 30]:
            if f'revenue_d{day}' in df.columns:
                feature_name = f'arpi_d{day}'
                df[feature_name] = df[f'revenue_d{day}'] / df['installs'].clip(lower=1)
                self.features_created.append(feature_name)
        
        # ARPU (Average Revenue Per User) - using retained users
        for day in [1, 7, 30]:
            revenue_col = f'revenue_d{day}'
            retained_col = f'retained_users_d{day}'
            if revenue_col in df.columns and retained_col in df.columns:
                feature_name = f'arpu_d{day}'
                df[feature_name] = df[revenue_col] / df[retained_col].clip(lower=1)
                self.features_created.append(feature_name)
        
        # Revenue growth rates
        revenue_pairs = [(1, 7), (7, 30), (1, 30)]
        for d1, d2 in revenue_pairs:
            if all(f'revenue_d{d}' in df.columns for d in [d1, d2]):
                feature_name = f'revenue_growth_d{d1}_d{d2}'
                df[feature_name] = (
                    df[f'revenue_d{d2}'] / df[f'revenue_d{d1}'].clip(lower=0.01)
                ).clip(upper=50)
                self.features_created.append(feature_name)
        
        # Revenue acceleration
        if all(f'revenue_growth_d{p[0]}_d{p[1]}' in df.columns for p in [(1, 7), (7, 30)]):
            df['revenue_acceleration'] = (
                df['revenue_growth_d7_d30'] / df['revenue_growth_d1_d7'].clip(lower=0.1)
            ).clip(0.1, 10)
            self.features_created.append('revenue_acceleration')
        
        # Early monetization strength
        if all(f'revenue_d{d}' in df.columns for d in [7, 30]):
            df['early_monetization_ratio'] = (
                df['revenue_d7'] / df['revenue_d30'].clip(lower=0.01)
            )
            self.features_created.append('early_monetization_ratio')
        
        logger.info(f"Created {len([f for f in self.features_created if 'revenue' in f or 'arpi' in f or 'arpu' in f])} revenue features")
        return df
    
    def create_monetization_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create monetization and payer features.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with monetization features added
        """
        # Payer conversion rates
        for day in self.PAYER_DAYS:
            payer_col = f'payers_d{day}'
            if payer_col in df.columns:
                # Conversion rate
                feature_name = f'payer_conversion_d{day}'
                df[feature_name] = df[payer_col] / df['installs'].clip(lower=1)
                self.features_created.append(feature_name)
                
                # ARPPU (Average Revenue Per Paying User)
                revenue_col = f'revenue_d{day}'
                if revenue_col in df.columns:
                    arppu_name = f'arppu_d{day}'
                    df[arppu_name] = df[revenue_col] / df[payer_col].clip(lower=1)
                    self.features_created.append(arppu_name)
        
        # IAP vs Ad revenue split
        if all(col in df.columns for col in ['iap_revenue_d30', 'ad_revenue_d30']):
            total_rev = (df['iap_revenue_d30'] + df['ad_revenue_d30']).clip(lower=0.01)
            
            df['iap_revenue_share'] = df['iap_revenue_d30'] / total_rev
            df['ad_revenue_share'] = df['ad_revenue_d30'] / total_rev
            
            # Dominant monetization type
            df['is_iap_dominant'] = (df['iap_revenue_share'] > 0.7).astype(int)
            df['is_ad_dominant'] = (df['ad_revenue_share'] > 0.7).astype(int)
            df['is_hybrid_monetization'] = (
                (df['iap_revenue_share'] > 0.3) & (df['ad_revenue_share'] > 0.3)
            ).astype(int)
            
            self.features_created.extend([
                'iap_revenue_share', 'ad_revenue_share',
                'is_iap_dominant', 'is_ad_dominant', 'is_hybrid_monetization'
            ])
        
        logger.info(f"Created {len([f for f in self.features_created if 'payer' in f or 'monetization' in f or 'iap' in f or 'ad' in f])} monetization features")
        return df
    
    def create_engagement_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create engagement and session features.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with engagement features added
        """
        # Sessions per user
        for day in self.SESSION_DAYS:
            session_col = f'sessions_d{day}'
            if session_col in df.columns:
                # Per install
                feature_name = f'sessions_per_install_d{day}'
                df[feature_name] = df[session_col] / df['installs'].clip(lower=1)
                self.features_created.append(feature_name)
                
                # Per retained user
                retained_col = f'retained_users_d{day}'
                if retained_col in df.columns:
                    feature_name = f'sessions_per_retained_d{day}'
                    df[feature_name] = df[session_col] / df[retained_col].clip(lower=1)
                    self.features_created.append(feature_name)
        
        # Session growth
        if all(f'sessions_d{d}' in df.columns for d in [1, 7]):
            df['session_growth_d1_d7'] = (
                df['sessions_d7'] / df['sessions_d1'].clip(lower=0.01)
            ).clip(upper=20)
            self.features_created.append('session_growth_d1_d7')
        
        # Revenue per session (monetization efficiency)
        for day in [1, 7]:
            revenue_col = f'revenue_d{day}'
            session_col = f'sessions_d{day}'
            if all(col in df.columns for col in [revenue_col, session_col]):
                feature_name = f'revenue_per_session_d{day}'
                df[feature_name] = df[revenue_col] / df[session_col].clip(lower=1)
                self.features_created.append(feature_name)
        
        logger.info(f"Created {len([f for f in self.features_created if 'session' in f or 'engagement' in f])} engagement features")
        return df
    
    def create_efficiency_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create ROI and efficiency features.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with efficiency features added
        """
        # CPI (Cost Per Install)
        df['cpi'] = df['network_cost'] / df['installs'].clip(lower=1)
        self.features_created.append('cpi')
        
        # ROAS (Return on Ad Spend) at different time periods
        for day in [1, 7, 14, 30]:
            if f'revenue_d{day}' in df.columns:
                feature_name = f'roas_d{day}'
                df[feature_name] = df[f'revenue_d{day}'] / df['network_cost'].clip(lower=0.01)
                self.features_created.append(feature_name)
        
        # Cost per retained user
        for day in [7, 30]:
            retained_col = f'retained_users_d{day}'
            if retained_col in df.columns:
                feature_name = f'cost_per_retained_d{day}'
                df[feature_name] = df['network_cost'] / df[retained_col].clip(lower=1)
                self.features_created.append(feature_name)
        
        # Payback indicators
        if 'roas_d30' in df.columns:
            df['is_profitable_d30'] = (df['roas_d30'] > 1.0).astype(int)
            self.features_created.append('is_profitable_d30')
        
        logger.info(f"Created {len([f for f in self.features_created if 'roas' in f or 'cost' in f or 'cpi' in f])} efficiency features")
        return df
    
    def create_categorical_encodings(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create encodings for categorical variables.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with categorical encodings added
        """
        categorical_cols = ['game_genre', 'os_name', 'country_code', 'channel']
        
        for col in categorical_cols:
            if col in df.columns:
                # Frequency encoding
                freq_map = df[col].value_counts().to_dict()
                df[f'{col}_frequency'] = df[col].map(freq_map)
                self.features_created.append(f'{col}_frequency')
                
                # High-frequency indicator (top 20%)
                threshold = df[f'{col}_frequency'].quantile(0.8)
                df[f'{col}_is_common'] = (df[f'{col}_frequency'] > threshold).astype(int)
                self.features_created.append(f'{col}_is_common')
        
        # Platform-specific features
        if 'os_name' in df.columns:
            df['is_ios'] = (df['os_name'] == 'iOS').astype(int)
            df['is_android'] = (df['os_name'] == 'Android').astype(int)
            self.features_created.extend(['is_ios', 'is_android'])
        
        # Geographic features
        if 'country_code' in df.columns:
            # Tier 1 countries (high-value markets)
            tier1_countries = ['US', 'GB', 'CA', 'AU', 'DE', 'FR', 'JP']
            df['is_tier1_country'] = df['country_code'].isin(tier1_countries).astype(int)
            self.features_created.append('is_tier1_country')
        
        logger.info(f"Created {len([f for f in self.features_created if 'frequency' in f or 'is_' in f])} categorical features")
        return df
    
    def create_composite_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create composite scores combining multiple features.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with composite scores added
        """
        # Early performance score (D1-D7 metrics)
        early_features = []
        weights = []
        
        if 'retention_rate_d7' in df.columns:
            early_features.append(df['retention_rate_d7'])
            weights.append(0.3)
        
        if 'arpi_d7' in df.columns:
            # Normalize ARPI to 0-1 scale
            arpi_max = df['arpi_d7'].quantile(0.95)
            if arpi_max > 0:
                arpi_norm = (df['arpi_d7'] / arpi_max).clip(lower=0, upper=1)
            else:
                arpi_norm = df['arpi_d7']
            early_features.append(arpi_norm)
            weights.append(0.4)
        
        if 'roas_d7' in df.columns:
            # Normalize ROAS to 0-1 scale
            roas_norm = (df['roas_d7'] / 2.0).clip(lower=0, upper=1)  # Assume ROAS of 2 is excellent
            early_features.append(roas_norm)
            weights.append(0.3)
        
        if early_features:
            # Weighted average
            total_weight = sum(weights)
            weights = [w/total_weight for w in weights]
            df['early_performance_score'] = sum(f * w for f, w in zip(early_features, weights))
            self.features_created.append('early_performance_score')
        
        # Cohort quality score
        quality_features = []
        weights = []
        
        if 'retention_rate_d30' in df.columns:
            quality_features.append(df['retention_rate_d30'])
            weights.append(0.4)
        
        if 'payer_conversion_d30' in df.columns:
            quality_features.append(df['payer_conversion_d30'])
            weights.append(0.3)
        
        if 'sessions_per_retained_d7' in df.columns:
            # Normalize sessions
            sessions_max = df['sessions_per_retained_d7'].quantile(0.95)
            if sessions_max > 0:
                sessions_norm = (df['sessions_per_retained_d7'] / sessions_max).clip(lower=0, upper=1)
            else:
                sessions_norm = df['sessions_per_retained_d7']
            quality_features.append(sessions_norm)
            weights.append(0.3)
        
        if quality_features:
            total_weight = sum(weights)
            weights = [w/total_weight for w in weights]
            df['cohort_quality_score'] = sum(f * w for f, w in zip(quality_features, weights))
            self.features_created.append('cohort_quality_score')
        
        logger.info(f"Created {len([f for f in self.features_created if 'score' in f])} composite scores")
        return df
    
    def get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """
        Get list of feature columns for modeling.
        
        Args:
            df: DataFrame with all features
            
        Returns:
            List of feature column names
        """
        # Start with created features
        feature_cols = list(set(self.features_created))
        
        # Add basic metrics
        basic_metrics = ['installs', 'network_cost']
        feature_cols.extend([col for col in basic_metrics if col in df.columns])
        
        # Ensure all features exist in dataframe
        feature_cols = [col for col in feature_cols if col in df.columns]
        
        # Remove any non-numeric columns that weren't properly encoded
        numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        feature_cols = [
            col for col in feature_cols 
            if df[col].dtype in numeric_dtypes or pd.api.types.is_numeric_dtype(df[col])
        ]
        
        return sorted(feature_cols)
    
    def run(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], Dict[str, Any]]:
        """
        Execute complete feature engineering pipeline.
        
        Args:
            df: Input DataFrame with preprocessed data
            
        Returns:
            Tuple of (DataFrame with features, feature list, statistics)
        """
        logger.info("Starting feature engineering")
        initial_cols = df.shape[1]
        
        # Create all feature types
        df = self.create_retention_features(df)
        df = self.create_revenue_features(df)
        df = self.create_monetization_features(df)
        df = self.create_engagement_features(df)
        df = self.create_efficiency_features(df)
        df = self.create_categorical_encodings(df)
        df = self.create_composite_scores(df)
        
        # Clean up features
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Get feature columns
        feature_cols = self.get_feature_columns(df)
        
        # Fill NaN values in features
        for col in feature_cols:
            if df[col].isnull().any():
                df[col].fillna(0, inplace=True)
        
        # Calculate statistics
        stats = {
            'initial_columns': initial_cols,
            'features_created': len(self.features_created),
            'final_feature_count': len(feature_cols),
            'feature_groups': {
                'retention': len([f for f in feature_cols if 'retention' in f]),
                'revenue': len([f for f in feature_cols if 'revenue' in f or 'arpi' in f or 'arpu' in f]),
                'monetization': len([f for f in feature_cols if 'payer' in f or 'monetization' in f]),
                'engagement': len([f for f in feature_cols if 'session' in f]),
                'efficiency': len([f for f in feature_cols if 'roas' in f or 'cost' in f or 'cpi' in f]),
                'categorical': len([f for f in feature_cols if 'frequency' in f or 'is_' in f]),
                'composite': len([f for f in feature_cols if 'score' in f])
            }
        }
        
        logger.info(f"Feature engineering completed: {initial_cols} → {len(feature_cols)} features")
        return df, feature_cols, stats