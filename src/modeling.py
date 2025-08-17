"""Modeling module for LTV prediction pipeline."""

import logging
from typing import Dict, Any, Tuple, List, Optional
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor

logger = logging.getLogger(__name__)


class LTVModeler:
    """Train and evaluate LTV prediction models."""
    
    # Default model configuration
    DEFAULT_MODEL_PARAMS = {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'random_state': 42,
        'n_jobs': -1,
        'verbosity': 0
    }
    
    def __init__(
        self,
        target_day: int = 90,
        test_size: float = 0.2,
        cv_folds: int = 5,
        random_state: int = 42,
        model_params: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize LTV modeler.
        
        Args:
            target_day: Day to predict revenue for (e.g., 90 for D90)
            test_size: Proportion of data for testing
            cv_folds: Number of cross-validation folds
            random_state: Random seed for reproducibility
            model_params: Optional XGBoost parameters override
        """
        self.target_column = f'revenue_d{target_day}'
        self.test_size = test_size
        self.cv_folds = cv_folds
        self.random_state = random_state
        
        # Model configuration
        self.model_params = self.DEFAULT_MODEL_PARAMS.copy()
        self.model_params['random_state'] = random_state
        if model_params:
            self.model_params.update(model_params)
        
        # Will be populated during training
        self.model = None
        self.feature_names = []
        self.label_encoders = {}
        self.feature_importance = {}
        self.metrics = {}
    
    def prepare_data(
        self,
        df: pd.DataFrame,
        feature_cols: List[str]
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare data for modeling.
        
        Args:
            df: DataFrame with features and target
            feature_cols: List of feature column names
            
        Returns:
            Tuple of (features DataFrame, target Series)
            
        Raises:
            ValueError: If target column not found or insufficient data
        """
        if self.target_column not in df.columns:
            raise ValueError(f"Target column {self.target_column} not found")
        
        # Remove potential data leakage features (anything after D30)
        leakage_keywords = ['d60', 'd90', 'd45', 'd120', 'd180']
        safe_features = [
            f for f in feature_cols
            if not any(keyword in f.lower() for keyword in leakage_keywords)
        ]
        
        if len(safe_features) < len(feature_cols):
            logger.warning(
                f"Removed {len(feature_cols) - len(safe_features)} potential leakage features"
            )
        
        # Select features and target
        X = df[safe_features].copy()
        y = df[self.target_column].copy()
        
        # Remove rows with missing target
        valid_mask = y.notna()
        X = X[valid_mask]
        y = y[valid_mask]
        
        # Ensure non-negative target
        y = y.clip(lower=0)
        
        # Store feature names
        self.feature_names = list(X.columns)
        
        logger.info(
            f"Prepared {len(X)} samples with {len(self.feature_names)} features. "
            f"Target range: ${y.min():.2f} - ${y.max():.2f}"
        )
        
        return X, y
    
    def encode_categoricals(self, X: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Encode categorical features for XGBoost.
        
        Args:
            X: Features DataFrame
            fit: Whether to fit encoders (True for training)
            
        Returns:
            DataFrame with encoded features
        """
        X_encoded = X.copy()
        categorical_cols = X_encoded.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            if fit:
                # Fit and transform
                le = LabelEncoder()
                X_encoded[col] = X_encoded[col].fillna('unknown')
                X_encoded[col] = le.fit_transform(X_encoded[col])
                self.label_encoders[col] = le
            else:
                # Transform only
                if col in self.label_encoders:
                    le = self.label_encoders[col]
                    X_encoded[col] = X_encoded[col].fillna('unknown')
                    # Handle unknown categories
                    X_encoded[col] = X_encoded[col].apply(
                        lambda x: le.transform([x])[0] 
                        if x in le.classes_ else len(le.classes_)
                    )
        
        return X_encoded
    
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None
    ) -> XGBRegressor:
        """
        Train XGBoost model.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Optional validation features
            y_val: Optional validation target
            
        Returns:
            Trained XGBoost model
        """
        # Configure model with early stopping if validation data provided
        model_params = self.model_params.copy()
        if X_val is not None and y_val is not None:
            model_params['early_stopping_rounds'] = 10
        
        model = XGBRegressor(**model_params)
        
        # Setup fit parameters
        fit_params = {}
        if X_val is not None and y_val is not None:
            fit_params['eval_set'] = [(X_val, y_val)]
            fit_params['verbose'] = False
        
        # Train model
        model.fit(X_train, y_train, **fit_params)
        
        logger.info(f"Model trained with {len(X_train)} samples")
        return model
    
    def evaluate(
        self,
        model: XGBRegressor,
        X: pd.DataFrame,
        y: pd.Series,
        dataset_name: str = "test"
    ) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            model: Trained model
            X: Features
            y: True target values
            dataset_name: Name for logging
            
        Returns:
            Dictionary of metrics
        """
        y_pred = model.predict(X)
        
        # Calculate metrics
        metrics = {
            'mae': mean_absolute_error(y, y_pred),
            'rmse': np.sqrt(mean_squared_error(y, y_pred)),
            'r2': r2_score(y, y_pred),
            'mape': np.mean(np.abs((y - y_pred) / y.clip(lower=1))) * 100
        }
        
        # Business metrics
        metrics['total_actual'] = y.sum()
        metrics['total_predicted'] = y_pred.sum()
        metrics['total_error_pct'] = abs(
            metrics['total_predicted'] - metrics['total_actual']
        ) / metrics['total_actual'] * 100
        
        logger.info(
            f"{dataset_name} metrics: "
            f"RMSE=${metrics['rmse']:.2f}, "
            f"R²={metrics['r2']:.3f}, "
            f"MAPE={metrics['mape']:.1f}%"
        )
        
        return metrics
    
    def cross_validate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """
        Perform cross-validation.
        
        Args:
            X: Features
            y: Target
            
        Returns:
            Cross-validation results
        """
        model = XGBRegressor(**self.model_params)
        kfold = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        
        # Calculate RMSE scores
        neg_mse_scores = cross_val_score(
            model, X, y, cv=kfold, 
            scoring='neg_mean_squared_error', 
            n_jobs=-1
        )
        rmse_scores = np.sqrt(-neg_mse_scores)
        
        # Calculate R² scores
        r2_scores = cross_val_score(
            model, X, y, cv=kfold,
            scoring='r2',
            n_jobs=-1
        )
        
        cv_results = {
            'rmse_mean': rmse_scores.mean(),
            'rmse_std': rmse_scores.std(),
            'r2_mean': r2_scores.mean(),
            'r2_std': r2_scores.std()
        }
        
        logger.info(
            f"CV Results: RMSE=${cv_results['rmse_mean']:.2f}±{cv_results['rmse_std']:.2f}, "
            f"R²={cv_results['r2_mean']:.3f}±{cv_results['r2_std']:.3f}"
        )
        
        return cv_results
    
    def extract_feature_importance(self, model: XGBRegressor) -> Dict[str, float]:
        """
        Extract feature importance from model.
        
        Args:
            model: Trained XGBoost model
            
        Returns:
            Dictionary of feature importance scores
        """
        importance_scores = model.feature_importances_
        importance_dict = dict(zip(self.feature_names, importance_scores))
        
        # Sort by importance
        sorted_importance = dict(
            sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        )
        
        # Log top features
        top_features = list(sorted_importance.items())[:10]
        logger.info("Top 10 features:")
        for i, (feat, imp) in enumerate(top_features, 1):
            logger.info(f"  {i}. {feat}: {imp:.4f}")
        
        return sorted_importance
    
    def save_model(self, model: XGBRegressor, output_dir: Path) -> None:
        """
        Save model and metadata.
        
        Args:
            model: Trained model
            output_dir: Directory to save model
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = output_dir / "model.joblib"
        joblib.dump(model, model_path)
        
        # Convert numpy types to Python native types for JSON serialization
        def convert_to_native(obj):
            """Convert numpy types to native Python types."""
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_native(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_native(v) for v in obj]
            return obj
        
        # Save metadata
        import json
        metadata = {
            'target_column': self.target_column,
            'feature_names': self.feature_names,
            'feature_importance': convert_to_native(self.feature_importance),
            'model_params': convert_to_native(self.model_params),
            'metrics': convert_to_native(self.metrics),
            'label_encoders': {
                col: list(le.classes_) 
                for col, le in self.label_encoders.items()
            }
        }
        
        metadata_path = output_dir / "model_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Model saved to {output_dir}")
    
    def run(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        output_dir: Optional[Path] = None
    ) -> Tuple[XGBRegressor, Dict[str, Any]]:
        """
        Execute complete modeling pipeline.
        
        Args:
            df: DataFrame with features and target
            feature_cols: List of feature column names
            output_dir: Optional directory to save model
            
        Returns:
            Tuple of (trained model, metrics dictionary)
        """
        logger.info("Starting modeling pipeline")
        
        # Prepare data
        X, y = self.prepare_data(df, feature_cols)
        
        # Encode categoricals
        X = self.encode_categoricals(X, fit=True)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        
        # Further split for validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=self.random_state
        )
        
        logger.info(
            f"Data split: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}"
        )
        
        # Train model
        self.model = self.train(X_train, y_train, X_val, y_val)
        
        # Evaluate
        train_metrics = self.evaluate(self.model, X_train, y_train, "Train")
        test_metrics = self.evaluate(self.model, X_test, y_test, "Test")
        
        # Cross-validation
        cv_results = self.cross_validate(X, y)
        
        # Feature importance
        self.feature_importance = self.extract_feature_importance(self.model)
        
        # Compile metrics
        self.metrics = {
            'train': train_metrics,
            'test': test_metrics,
            'cv': cv_results,
            'data': {
                'total_samples': len(X),
                'train_samples': len(X_train),
                'test_samples': len(X_test),
                'n_features': len(self.feature_names)
            }
        }
        
        # Save if output directory provided
        if output_dir:
            self.save_model(self.model, output_dir)
        
        logger.info("Modeling pipeline completed")
        return self.model, self.metrics