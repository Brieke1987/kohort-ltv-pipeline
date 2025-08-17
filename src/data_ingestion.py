"""Data ingestion module for LTV pipeline."""

import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class DataIngestion:
    """Handle data loading and validation for LTV pipeline."""
    
    # Class-level constants
    EXPECTED_DTYPES = {
        'day': 'object',
        'game_id': 'object',
        'game_genre': 'object',
        'os_name': 'object',
        'country_code': 'object',
        'channel': 'object',
        'campaign_id': 'object',
        'installs': 'int64',
        'network_cost': 'float64',
        'retained_users_d1': 'int64',
        'retained_users_d3': 'int64',
        'retained_users_d7': 'int64',
        'retained_users_d14': 'int64',
        'retained_users_d30': 'int64',
        'revenue_d1': 'float64',
        'revenue_d3': 'float64',
        'revenue_d7': 'float64',
        'revenue_d14': 'float64',
        'revenue_d30': 'float64',
        'revenue_d60': 'float64',
        'revenue_d90': 'float64',
        'iap_revenue_d30': 'float64',
        'ad_revenue_d30': 'float64',
        'sessions_d1': 'float64',
        'sessions_d7': 'float64',
        'payers_d7': 'int64',
        'payers_d30': 'int64'
    }
    
    CATEGORICAL_COLS = ['game_genre', 'os_name', 'country_code', 'channel']
    KEY_METRICS = ['installs', 'network_cost', 'revenue_d30', 'revenue_d90']
    
    def __init__(
        self,
        max_missing_rate: float = 0.05,
        min_records: int = 1000,
        max_duplicate_rate: float = 0.02  # 2% is more reasonable for real-world data
    ):
        """
        Initialize data ingestion with quality thresholds.
        
        Args:
            max_missing_rate: Maximum acceptable missing data rate
            min_records: Minimum required number of records
            max_duplicate_rate: Maximum acceptable duplicate rate
        """
        self.max_missing_rate = max_missing_rate
        self.min_records = min_records
        self.max_duplicate_rate = max_duplicate_rate
        
    def load_data(self, file_path: Path) -> pd.DataFrame:
        """
        Load parquet file with validation.
        
        Args:
            file_path: Path to the parquet file
            
        Returns:
            Loaded DataFrame
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file is empty or corrupted
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        logger.info(f"Loading data from {file_path}")
        df = pd.read_parquet(file_path)
        
        if df.empty:
            raise ValueError(f"Loaded DataFrame is empty: {file_path}")
        
        logger.info(f"Successfully loaded {len(df):,} records")
        return df
    
    def validate_schema(self, df: pd.DataFrame) -> None:
        """
        Validate dataframe schema against expected structure.
        
        Args:
            df: DataFrame to validate
            
        Raises:
            ValueError: If required columns are missing
        """
        missing_cols = set(self.EXPECTED_DTYPES.keys()) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        logger.info("Schema validation passed")
    
    def calculate_quality_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate comprehensive data quality metrics.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dictionary containing quality metrics
        """
        quality_report = {
            'total_records': len(df),
            'total_columns': len(df.columns),
            'missing_rates': {},
            'duplicates': {
                'count': df.duplicated().sum(),
                'rate': df.duplicated().sum() / len(df)
            }
        }
        
        # Calculate missing rates
        for col in df.columns:
            missing_rate = df[col].isnull().sum() / len(df)
            if missing_rate > 0:
                quality_report['missing_rates'][col] = missing_rate
        
        # Date range analysis
        if 'day' in df.columns:
            df['day_parsed'] = pd.to_datetime(df['day'], errors='coerce')
            valid_dates = df['day_parsed'].notna()
            if valid_dates.any():
                min_date = df.loc[valid_dates, 'day_parsed'].min()
                max_date = df.loc[valid_dates, 'day_parsed'].max()
                quality_report['date_range'] = {
                    'min': str(min_date.date()),
                    'max': str(max_date.date()),
                    'span_days': (max_date - min_date).days
                }
            df.drop('day_parsed', axis=1, inplace=True)
        
        # Categorical summary
        quality_report['categorical'] = {
            col: df[col].nunique() 
            for col in self.CATEGORICAL_COLS 
            if col in df.columns
        }
        
        # Numerical summary
        quality_report['numerical'] = {}
        for col in self.KEY_METRICS:
            if col in df.columns:
                col_data = df[col].dropna()
                if not col_data.empty:
                    quality_report['numerical'][col] = {
                        'mean': col_data.mean(),
                        'median': col_data.median(),
                        'min': col_data.min(),
                        'max': col_data.max(),
                        'negative_count': (col_data < 0).sum()
                    }
        
        return quality_report
    
    def validate_quality(self, quality_report: Dict[str, Any]) -> None:
        """
        Validate data quality against thresholds.
        
        Args:
            quality_report: Quality metrics dictionary
            
        Raises:
            ValueError: If quality thresholds are violated
        """
        issues = []
        
        # Check record count
        if quality_report['total_records'] < self.min_records:
            issues.append(
                f"Insufficient records: {quality_report['total_records']} < {self.min_records}"
            )
        
        # Check duplicate rate
        if quality_report['duplicates']['rate'] > self.max_duplicate_rate:
            issues.append(
                f"High duplicate rate: {quality_report['duplicates']['rate']:.1%} > "
                f"{self.max_duplicate_rate:.1%}"
            )
        
        # Check missing rates
        high_missing = [
            f"{col} ({rate:.1%})" 
            for col, rate in quality_report['missing_rates'].items()
            if rate > self.max_missing_rate
        ]
        if high_missing:
            issues.append(f"High missing rates: {', '.join(high_missing)}")
        
        if issues:
            for issue in issues:
                logger.warning(f"Quality issue: {issue}")
            raise ValueError(f"Data quality validation failed: {'; '.join(issues)}")
        
        logger.info("Data quality validation passed")
    
    def log_summary(self, quality_report: Dict[str, Any]) -> None:
        """
        Log summary statistics.
        
        Args:
            quality_report: Quality metrics dictionary
        """
        logger.info(f"Records: {quality_report['total_records']:,}")
        
        if 'date_range' in quality_report:
            dr = quality_report['date_range']
            logger.info(f"Date range: {dr['min']} to {dr['max']} ({dr['span_days']} days)")
        
        if quality_report['categorical']:
            logger.info(f"Unique values: {quality_report['categorical']}")
        
        for metric, stats in quality_report.get('numerical', {}).items():
            logger.info(
                f"{metric}: mean={stats['mean']:.2f}, "
                f"range=[{stats['min']:.2f}, {stats['max']:.2f}]"
            )
    
    def run(self, file_path: Path) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Execute complete data ingestion pipeline.
        
        Args:
            file_path: Path to input parquet file
            
        Returns:
            Tuple of (DataFrame, quality_report)
            
        Raises:
            Various exceptions for data quality issues
        """
        logger.info("Starting data ingestion pipeline")
        
        # Load and validate
        df = self.load_data(file_path)
        self.validate_schema(df)
        
        # Calculate quality metrics
        quality_report = self.calculate_quality_metrics(df)
        
        # Validate quality
        self.validate_quality(quality_report)
        
        # Log summary
        self.log_summary(quality_report)
        
        logger.info("Data ingestion completed successfully")
        return df, quality_report