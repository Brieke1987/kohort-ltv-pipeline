"""Data preprocessing module for LTV pipeline."""

import logging
from typing import Dict, Any, Tuple, Optional
from pathlib import Path

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Handle data cleaning and preprocessing for LTV pipeline."""
    
    # Class-level constants for mappings
    CHANNEL_MAPPING = {
        'fb': 'facebook',
        'meta': 'facebook',
        'googleads': 'google',
        'google_ads': 'google',
        'unityads': 'unity',
        'unity_ads': 'unity',
        'ironsource': 'ironsource'
    }
    
    COUNTRY_MAPPING = {
        'USA': 'US',
        'UK': 'GB',
        'UNITED KINGDOM': 'GB'
    }
    
    # Columns that should never be negative
    NON_NEGATIVE_COLS = [
        'installs', 'network_cost', 'retained_users_d1', 'retained_users_d3',
        'retained_users_d7', 'retained_users_d14', 'retained_users_d30',
        'revenue_d1', 'revenue_d3', 'revenue_d7', 'revenue_d14', 'revenue_d30',
        'revenue_d60', 'revenue_d90', 'iap_revenue_d30', 'ad_revenue_d30',
        'sessions_d1', 'sessions_d7', 'payers_d7', 'payers_d30'
    ]
    
    def __init__(self, cap_outliers_percentile: float = 0.99):
        """
        Initialize preprocessor.
        
        Args:
            cap_outliers_percentile: Percentile for capping outliers (default 99th)
        """
        self.cap_percentile = cap_outliers_percentile
        self.stats = {}
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values with appropriate strategies.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with missing values handled
        """
        df = df.copy()
        missing_stats = {}
        
        # Campaign ID: Fill with 'unknown'
        if df['campaign_id'].isnull().any():
            count = df['campaign_id'].isnull().sum()
            df['campaign_id'].fillna('unknown', inplace=True)
            missing_stats['campaign_id'] = count
            logger.info(f"Filled {count} missing campaign_ids")
        
        # Network cost: Fill with channel median, then global median
        if df['network_cost'].isnull().any():
            count = df['network_cost'].isnull().sum()
            channel_medians = df.groupby('channel')['network_cost'].transform('median')
            df['network_cost'].fillna(channel_medians, inplace=True)
            df['network_cost'].fillna(df['network_cost'].median(), inplace=True)
            missing_stats['network_cost'] = count
            logger.info(f"Filled {count} missing network_costs")
        
        # Fill numeric columns with 0 (conservative approach for metrics)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().any():
                count = df[col].isnull().sum()
                df[col].fillna(0, inplace=True)
                missing_stats[col] = count
        
        self.stats['missing_values'] = missing_stats
        return df
    
    def standardize_text_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize text fields for consistency.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with standardized text fields
        """
        df = df.copy()
        
        # Standardize channels
        df['channel'] = (df['channel']
                        .str.lower()
                        .str.strip()
                        .str.replace(r'[^a-z0-9_]', '', regex=True))
        
        # Apply channel mapping
        for old, new in self.CHANNEL_MAPPING.items():
            df.loc[df['channel'] == old, 'channel'] = new
        
        # Standardize country codes
        df['country_code'] = df['country_code'].str.upper().str.strip()
        
        # Apply country mapping
        for old, new in self.COUNTRY_MAPPING.items():
            df.loc[df['country_code'] == old, 'country_code'] = new
        
        # Validate country codes (2 letters)
        invalid_mask = ~df['country_code'].str.match(r'^[A-Z]{2}$', na=False)
        if invalid_mask.any():
            df.loc[invalid_mask, 'country_code'] = 'XX'
            logger.warning(f"Fixed {invalid_mask.sum()} invalid country codes")
        
        # Standardize OS names
        df['os_name'] = df['os_name'].str.lower()
        df.loc[df['os_name'].str.contains('ios|iphone|ipad', na=False), 'os_name'] = 'iOS'
        df.loc[df['os_name'].str.contains('android', na=False), 'os_name'] = 'Android'
        
        # Standardize game genre
        df['game_genre'] = df['game_genre'].str.title().str.strip()
        
        # Parse dates
        df['day'] = pd.to_datetime(df['day'], errors='coerce')
        
        self.stats['unique_values'] = {
            'channels': df['channel'].nunique(),
            'countries': df['country_code'].nunique(),
            'genres': df['game_genre'].nunique(),
            'os': df['os_name'].nunique()
        }
        
        return df
    
    def fix_data_inconsistencies(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fix logical inconsistencies in the data.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with inconsistencies fixed
        """
        df = df.copy()
        fixes = {}
        
        # Ensure non-negative values
        for col in self.NON_NEGATIVE_COLS:
            if col in df.columns:
                negative_count = (df[col] < 0).sum()
                if negative_count > 0:
                    df[col] = df[col].clip(lower=0)
                    fixes[f'{col}_negative'] = negative_count
        
        # Fix retention inconsistencies (retention should decrease over time)
        retention_cols = ['retained_users_d1', 'retained_users_d3', 'retained_users_d7', 
                         'retained_users_d14', 'retained_users_d30']
        
        for i in range(len(retention_cols) - 1):
            curr_col = retention_cols[i]
            next_col = retention_cols[i + 1]
            if curr_col in df.columns and next_col in df.columns:
                # Next period retention shouldn't exceed current period
                mask = df[next_col] > df[curr_col]
                if mask.any():
                    df.loc[mask, next_col] = df.loc[mask, curr_col]
                    fixes[f'{next_col}_exceeds_{curr_col}'] = mask.sum()
        
        # Retention can't exceed installs
        for col in retention_cols:
            if col in df.columns:
                mask = df[col] > df['installs']
                if mask.any():
                    df.loc[mask, col] = df.loc[mask, 'installs']
                    fixes[f'{col}_exceeds_installs'] = mask.sum()
        
        # Fix revenue inconsistencies (revenue should increase over time)
        revenue_cols = ['revenue_d1', 'revenue_d3', 'revenue_d7', 'revenue_d14', 
                       'revenue_d30', 'revenue_d60', 'revenue_d90']
        
        for i in range(len(revenue_cols) - 1):
            curr_col = revenue_cols[i]
            next_col = revenue_cols[i + 1]
            if curr_col in df.columns and next_col in df.columns:
                # Next period revenue shouldn't be less than current period
                mask = df[next_col] < df[curr_col]
                if mask.any():
                    df.loc[mask, next_col] = df.loc[mask, curr_col]
                    fixes[f'{next_col}_less_than_{curr_col}'] = mask.sum()
        
        # IAP + Ad revenue should equal total D30 revenue (with small tolerance)
        if all(col in df.columns for col in ['iap_revenue_d30', 'ad_revenue_d30', 'revenue_d30']):
            total_from_components = df['iap_revenue_d30'] + df['ad_revenue_d30']
            discrepancy = abs(total_from_components - df['revenue_d30'])
            mask = discrepancy > 0.01  # Small tolerance for floating point
            if mask.any():
                # Use the max to be conservative
                df.loc[mask, 'revenue_d30'] = np.maximum(
                    df.loc[mask, 'revenue_d30'],
                    total_from_components[mask]
                )
                fixes['revenue_component_mismatch'] = mask.sum()
        
        # Payers can't exceed retained users
        if 'payers_d7' in df.columns and 'retained_users_d7' in df.columns:
            mask = df['payers_d7'] > df['retained_users_d7']
            if mask.any():
                df.loc[mask, 'payers_d7'] = df.loc[mask, 'retained_users_d7']
                fixes['payers_d7_exceeds_retention'] = mask.sum()
        
        if 'payers_d30' in df.columns and 'retained_users_d30' in df.columns:
            mask = df['payers_d30'] > df['retained_users_d30']
            if mask.any():
                df.loc[mask, 'payers_d30'] = df.loc[mask, 'retained_users_d30']
                fixes['payers_d30_exceeds_retention'] = mask.sum()
        
        self.stats['fixes_applied'] = fixes
        if fixes:
            logger.info(f"Applied {len(fixes)} data consistency fixes")
        
        return df
    
    def cap_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cap extreme outliers to percentile values.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with outliers capped
        """
        df = df.copy()
        outlier_stats = {}
        
        # Define columns where outliers should be capped
        cap_columns = ['network_cost', 'revenue_d1', 'revenue_d3', 'revenue_d7',
                      'revenue_d14', 'revenue_d30', 'revenue_d60', 'revenue_d90',
                      'iap_revenue_d30', 'ad_revenue_d30', 'sessions_d1', 'sessions_d7']
        
        for col in cap_columns:
            if col in df.columns and df[col].notna().any():
                # Calculate percentile cap
                upper_cap = df[col].quantile(self.cap_percentile)
                
                # Count outliers
                outliers = df[col] > upper_cap
                if outliers.any():
                    df.loc[outliers, col] = upper_cap
                    outlier_stats[col] = {
                        'count': outliers.sum(),
                        'cap_value': upper_cap
                    }
        
        self.stats['outliers_capped'] = outlier_stats
        if outlier_stats:
            logger.info(f"Capped outliers in {len(outlier_stats)} columns")
        
        return df
    
    def run(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Execute complete preprocessing pipeline.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Tuple of (preprocessed DataFrame, preprocessing statistics)
        """
        logger.info("Starting preprocessing pipeline")
        initial_shape = df.shape
        
        # Execute preprocessing steps
        df = self.handle_missing_values(df)
        df = self.standardize_text_fields(df)
        df = self.fix_data_inconsistencies(df)
        df = self.cap_outliers(df)
        
        # Add summary statistics
        self.stats['summary'] = {
            'initial_shape': initial_shape,
            'final_shape': df.shape,
            'null_counts': df.isnull().sum().to_dict()
        }
        
        logger.info(f"Preprocessing completed. Shape: {initial_shape} → {df.shape}")
        return df, self.stats