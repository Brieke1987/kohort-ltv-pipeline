"""Drift detection and monitoring module for LTV pipeline."""

import logging
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


class DriftDetector:
    """Detect data drift and model performance degradation."""
    
    # Drift severity thresholds
    PSI_THRESHOLDS = {'low': 0.1, 'medium': 0.2, 'high': 0.25}
    KS_THRESHOLDS = {'low': 0.05, 'medium': 0.1, 'high': 0.2}
    
    def __init__(self, n_bins: int = 10):
        """
        Initialize drift detector.
        
        Args:
            n_bins: Number of bins for distribution comparison
        """
        self.n_bins = n_bins
        self.baseline_stats = None
        self.baseline_distributions = None
    
    def fit_baseline(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_col: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Calculate and store baseline statistics.
        
        Args:
            df: Baseline DataFrame
            feature_cols: List of feature columns to monitor
            target_col: Optional target column to monitor
            
        Returns:
            Dictionary of baseline statistics
        """
        logger.info(f"Calculating baseline statistics for {len(feature_cols)} features")
        
        self.baseline_stats = {
            'features': {},
            'target': None,
            'sample_size': len(df)
        }
        
        self.baseline_distributions = {
            'features': {},
            'target': None
        }
        
        # Calculate feature statistics
        for col in feature_cols:
            if col not in df.columns:
                continue
            
            data = df[col].dropna()
            if data.empty:
                continue
            
            if pd.api.types.is_numeric_dtype(data):
                # Numerical feature
                self.baseline_stats['features'][col] = {
                    'type': 'numerical',
                    'mean': data.mean(),
                    'std': data.std(),
                    'min': data.min(),
                    'max': data.max(),
                    'q25': data.quantile(0.25),
                    'q75': data.quantile(0.75)
                }
                self.baseline_distributions['features'][col] = self._bin_numerical(data)
            else:
                # Categorical feature
                value_counts = data.value_counts(normalize=True)
                self.baseline_stats['features'][col] = {
                    'type': 'categorical',
                    'n_unique': len(value_counts),
                    'top_categories': value_counts.head(10).to_dict()
                }
                self.baseline_distributions['features'][col] = value_counts.to_dict()
        
        # Calculate target statistics if provided
        if target_col and target_col in df.columns:
            target_data = df[target_col].dropna()
            if not target_data.empty:
                self.baseline_stats['target'] = {
                    'mean': target_data.mean(),
                    'std': target_data.std(),
                    'min': target_data.min(),
                    'max': target_data.max()
                }
                self.baseline_distributions['target'] = self._bin_numerical(target_data)
        
        logger.info(f"Baseline statistics calculated for {len(self.baseline_stats['features'])} features")
        return self.baseline_stats
    
    def _bin_numerical(self, data: pd.Series) -> np.ndarray:
        """
        Create histogram bins for numerical data.
        
        Args:
            data: Numerical data series
            
        Returns:
            Array of bin frequencies
        """
        # Use quantile-based binning for robustness
        try:
            _, bin_edges = pd.qcut(data, q=self.n_bins, retbins=True, duplicates='drop')
            hist, _ = np.histogram(data, bins=bin_edges)
            return hist / len(data)  # Normalize to probabilities
        except Exception:
            # Fallback to uniform binning if quantile fails
            hist, _ = np.histogram(data, bins=self.n_bins)
            return hist / len(data)
    
    def calculate_psi(self, expected: np.ndarray, actual: np.ndarray) -> float:
        """
        Calculate Population Stability Index.
        
        Args:
            expected: Expected (baseline) distribution
            actual: Actual (current) distribution
            
        Returns:
            PSI value
        """
        # Avoid division by zero
        expected = np.clip(expected, 1e-10, None)
        actual = np.clip(actual, 1e-10, None)
        
        # Normalize if needed
        expected = expected / expected.sum()
        actual = actual / actual.sum()
        
        # Calculate PSI
        psi = np.sum((actual - expected) * np.log(actual / expected))
        return float(psi)
    
    def calculate_ks_statistic(
        self,
        baseline_data: pd.Series,
        current_data: pd.Series
    ) -> Tuple[float, float]:
        """
        Calculate Kolmogorov-Smirnov test statistic.
        
        Args:
            baseline_data: Baseline data
            current_data: Current data
            
        Returns:
            Tuple of (KS statistic, p-value)
        """
        try:
            ks_stat, p_value = stats.ks_2samp(baseline_data, current_data)
            return float(ks_stat), float(p_value)
        except Exception:
            return 0.0, 1.0
    
    def detect_drift(
        self,
        current_df: pd.DataFrame,
        feature_cols: Optional[List[str]] = None,
        target_col: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Detect drift in current data compared to baseline.
        
        Args:
            current_df: Current DataFrame to check for drift
            feature_cols: Optional list of features (uses baseline if not provided)
            target_col: Optional target column
            
        Returns:
            Dictionary containing drift detection results
        """
        if self.baseline_stats is None:
            raise ValueError("No baseline fitted. Call fit_baseline() first.")
        
        # Use baseline features if not specified
        if feature_cols is None:
            feature_cols = list(self.baseline_stats['features'].keys())
        
        results = {
            'features': {},
            'target': None,
            'summary': {
                'n_features_checked': 0,
                'n_drifted_features': 0,
                'max_psi': 0,
                'drift_detected': False
            }
        }
        
        # Check each feature
        for col in feature_cols:
            if col not in current_df.columns or col not in self.baseline_stats['features']:
                continue
            
            current_data = current_df[col].dropna()
            if current_data.empty:
                continue
            
            baseline_info = self.baseline_stats['features'][col]
            drift_info = {'drift_detected': False, 'severity': 'none'}
            
            if baseline_info['type'] == 'numerical':
                # Calculate PSI
                current_dist = self._bin_numerical(current_data)
                baseline_dist = self.baseline_distributions['features'][col]
                
                # Ensure same shape
                min_len = min(len(current_dist), len(baseline_dist))
                psi = self.calculate_psi(
                    baseline_dist[:min_len],
                    current_dist[:min_len]
                )
                
                drift_info['psi'] = psi
                
                # Determine severity
                if psi > self.PSI_THRESHOLDS['high']:
                    drift_info['severity'] = 'high'
                elif psi > self.PSI_THRESHOLDS['medium']:
                    drift_info['severity'] = 'medium'
                elif psi > self.PSI_THRESHOLDS['low']:
                    drift_info['severity'] = 'low'
                
                drift_info['drift_detected'] = drift_info['severity'] != 'none'
                
                # Add statistical comparison
                drift_info['mean_shift'] = current_data.mean() - baseline_info['mean']
                drift_info['std_ratio'] = current_data.std() / baseline_info['std'] if baseline_info['std'] > 0 else 1.0
                
            else:  # Categorical
                # Check for new/missing categories
                current_categories = set(current_data.unique())
                baseline_categories = set(baseline_info['top_categories'].keys())
                
                drift_info['new_categories'] = list(current_categories - baseline_categories)
                drift_info['missing_categories'] = list(baseline_categories - current_categories)
                
                # Calculate distribution shift for common categories
                current_dist = current_data.value_counts(normalize=True)
                baseline_dist = self.baseline_distributions['features'][col]
                
                common_categories = list(set(current_dist.index) & set(baseline_dist.keys()))
                if common_categories:
                    current_probs = np.array([current_dist.get(cat, 0) for cat in common_categories])
                    baseline_probs = np.array([baseline_dist.get(cat, 0) for cat in common_categories])
                    
                    psi = self.calculate_psi(baseline_probs, current_probs)
                    drift_info['psi'] = psi
                    
                    # Determine severity
                    if psi > self.PSI_THRESHOLDS['medium'] or len(drift_info['new_categories']) > 3:
                        drift_info['severity'] = 'high'
                    elif psi > self.PSI_THRESHOLDS['low'] or len(drift_info['new_categories']) > 0:
                        drift_info['severity'] = 'medium'
                    
                    drift_info['drift_detected'] = drift_info['severity'] != 'none'
            
            results['features'][col] = drift_info
            results['summary']['n_features_checked'] += 1
            
            if drift_info['drift_detected']:
                results['summary']['n_drifted_features'] += 1
                results['summary']['max_psi'] = max(
                    results['summary']['max_psi'],
                    drift_info.get('psi', 0)
                )
        
        # Check target drift if specified
        if target_col and self.baseline_stats['target'] is not None:
            if target_col in current_df.columns:
                current_target = current_df[target_col].dropna()
                if not current_target.empty:
                    baseline_target = self.baseline_stats['target']
                    
                    # Calculate PSI
                    current_dist = self._bin_numerical(current_target)
                    baseline_dist = self.baseline_distributions['target']
                    
                    min_len = min(len(current_dist), len(baseline_dist))
                    psi = self.calculate_psi(
                        baseline_dist[:min_len],
                        current_dist[:min_len]
                    )
                    
                    results['target'] = {
                        'psi': psi,
                        'mean_shift': current_target.mean() - baseline_target['mean'],
                        'drift_detected': psi > self.PSI_THRESHOLDS['medium']
                    }
        
        # Overall drift assessment
        drift_rate = results['summary']['n_drifted_features'] / max(results['summary']['n_features_checked'], 1)
        results['summary']['drift_detected'] = (
            drift_rate > 0.2 or 
            results['summary']['max_psi'] > self.PSI_THRESHOLDS['high'] or
            (results['target'] and results['target']['drift_detected'])
        )
        results['summary']['drift_rate'] = drift_rate
        
        logger.info(
            f"Drift detection complete: {results['summary']['n_drifted_features']}/{results['summary']['n_features_checked']} "
            f"features drifted (max PSI: {results['summary']['max_psi']:.3f})"
        )
        
        return results
    
    def get_alerts(self, drift_results: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        Generate alerts based on drift detection results.
        
        Args:
            drift_results: Output from detect_drift()
            
        Returns:
            List of alert dictionaries
        """
        alerts = []
        
        # Check for high severity feature drift
        high_drift_features = [
            feat for feat, info in drift_results['features'].items()
            if info.get('severity') == 'high'
        ]
        
        if high_drift_features:
            alerts.append({
                'severity': 'HIGH',
                'message': f"High drift detected in {len(high_drift_features)} features",
                'features': high_drift_features[:5],  # Top 5
                'action': 'Consider model retraining'
            })
        
        # Check overall drift rate
        drift_rate = drift_results['summary']['drift_rate']
        if drift_rate > 0.3:
            alerts.append({
                'severity': 'HIGH',
                'message': f"{drift_rate:.0%} of features show drift",
                'action': 'Immediate model review recommended'
            })
        elif drift_rate > 0.15:
            alerts.append({
                'severity': 'MEDIUM',
                'message': f"{drift_rate:.0%} of features show drift",
                'action': 'Monitor closely and prepare for retraining'
            })
        
        # Target drift alert
        if drift_results['target'] and drift_results['target']['drift_detected']:
            alerts.append({
                'severity': 'HIGH',
                'message': f"Target variable drift detected (PSI: {drift_results['target']['psi']:.3f})",
                'action': 'Investigate business changes and retrain model'
            })
        
        return alerts


# Make class available for import
__all__ = ['DriftDetector']