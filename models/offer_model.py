# Databricks notebook source
# MAGIC %md
# MAGIC # Offer Prioritization Model
# MAGIC ML model for predicting offer priority scores

# COMMAND ----------

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import lightgbm as lgb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import json

# COMMAND ----------

@dataclass
class OfferCatalog:
    """Healthcare offer catalog with targeting rules"""
    
    offers: List[Dict] = field(default_factory=lambda: [
        {
            "offer_id": "PREV_001",
            "name": "Annual Wellness Visit Reminder",
            "category": "preventive_care",
            "base_score": 60,
            "targeting_rules": {
                "age_boost": {"min": 40, "boost": 10},
                "days_since_last_claim_boost": {"min": 180, "boost": 15},
                "has_chronic_condition_boost": 10
            }
        },
        {
            "offer_id": "PREV_002",
            "name": "Cancer Screening Program",
            "category": "preventive_care",
            "base_score": 55,
            "targeting_rules": {
                "age_boost": {"min": 50, "boost": 20},
                "high_risk_flag_boost": 15
            }
        },
        {
            "offer_id": "CHRON_001",
            "name": "Diabetes Management Program",
            "category": "chronic_disease",
            "base_score": 50,
            "targeting_rules": {
                "has_diabetes_boost": 40,
                "high_risk_flag_boost": 10,
                "pharmacy_utilization_boost": {"min": 0.3, "boost": 5}
            }
        },
        {
            "offer_id": "CHRON_002",
            "name": "Heart Health Program",
            "category": "chronic_disease",
            "base_score": 50,
            "targeting_rules": {
                "has_cardiovascular_boost": 40,
                "age_boost": {"min": 50, "boost": 10},
                "high_risk_flag_boost": 10
            }
        },
        {
            "offer_id": "CHRON_003",
            "name": "Respiratory Care Program",
            "category": "chronic_disease",
            "base_score": 50,
            "targeting_rules": {
                "has_respiratory_boost": 40,
                "is_allergy_season_boost": 5
            }
        },
        {
            "offer_id": "MH_001",
            "name": "Mental Health Support",
            "category": "mental_health",
            "base_score": 55,
            "targeting_rules": {
                "has_mental_health_boost": 35,
                "age_boost": {"max": 45, "boost": 10},
                "engagement_trend_boost": {"min": 0.5, "boost": 5}
            }
        },
        {
            "offer_id": "MH_002",
            "name": "Stress Management Workshop",
            "category": "mental_health",
            "base_score": 50,
            "targeting_rules": {
                "total_claims_count_boost": {"min": 10, "boost": 10},
                "avg_response_rate_boost": {"min": 0.2, "boost": 10}
            }
        },
        {
            "offer_id": "PHARM_001",
            "name": "Generic Drug Switch Program",
            "category": "pharmacy",
            "base_score": 45,
            "targeting_rules": {
                "pharmacy_utilization_rate_boost": {"min": 0.4, "boost": 20},
                "total_member_cost_boost": {"min": 500, "boost": 15}
            }
        },
        {
            "offer_id": "PHARM_002",
            "name": "Mail-Order Pharmacy",
            "category": "pharmacy",
            "base_score": 45,
            "targeting_rules": {
                "pharmacy_utilization_rate_boost": {"min": 0.3, "boost": 15},
                "has_chronic_condition_boost": 10
            }
        },
        {
            "offer_id": "TELE_001",
            "name": "Virtual Primary Care",
            "category": "telehealth",
            "base_score": 50,
            "targeting_rules": {
                "app_engagement_rate_boost": {"min": 0.2, "boost": 15},
                "primary_care_visit_count_boost": {"min": 2, "boost": 10}
            }
        },
        {
            "offer_id": "TELE_002",
            "name": "Virtual Specialist Consultations",
            "category": "telehealth",
            "base_score": 45,
            "targeting_rules": {
                "specialist_visit_count_boost": {"min": 2, "boost": 20},
                "is_complex_patient_boost": 15
            }
        },
        {
            "offer_id": "FIT_001",
            "name": "Gym Membership Discount",
            "category": "fitness",
            "base_score": 40,
            "targeting_rules": {
                "age_boost": {"max": 55, "boost": 10},
                "has_cardiovascular_boost": 10,
                "has_diabetes_boost": 10
            }
        },
        {
            "offer_id": "FIT_002",
            "name": "Nutrition Coaching",
            "category": "fitness",
            "base_score": 45,
            "targeting_rules": {
                "has_diabetes_boost": 15,
                "has_cardiovascular_boost": 10,
                "high_risk_flag_boost": 10
            }
        },
        {
            "offer_id": "NAV_001",
            "name": "Care Navigator Assignment",
            "category": "care_navigation",
            "base_score": 40,
            "targeting_rules": {
                "is_complex_patient_boost": 30,
                "total_claims_count_boost": {"min": 15, "boost": 15},
                "chronic_condition_count_boost": {"min": 2, "boost": 10}
            }
        },
        {
            "offer_id": "COST_001",
            "name": "HSA/FSA Optimization",
            "category": "cost_savings",
            "base_score": 45,
            "targeting_rules": {
                "remaining_deductible_pct_boost": {"min": 0.5, "boost": 15},
                "is_q4_boost": 20
            }
        },
        {
            "offer_id": "COST_002",
            "name": "In-Network Provider Finder",
            "category": "cost_savings",
            "base_score": 45,
            "targeting_rules": {
                "in_network_rate_boost": {"max": 0.8, "boost": 20},
                "total_member_cost_boost": {"min": 500, "boost": 10}
            }
        }
    ])
    
    def get_offer_ids(self) -> List[str]:
        """Get list of all offer IDs"""
        return [o["offer_id"] for o in self.offers]
    
    def get_offer_by_id(self, offer_id: str) -> Optional[Dict]:
        """Get offer details by ID"""
        for offer in self.offers:
            if offer["offer_id"] == offer_id:
                return offer
        return None


# COMMAND ----------

class RuleBasedScorer:
    """
    Generate target scores based on business rules.
    Used to create training labels from member features.
    """
    
    def __init__(self, catalog: Optional[OfferCatalog] = None):
        self.catalog = catalog or OfferCatalog()
        
    def generate_target_scores(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate offer priority scores for each member based on rules
        
        Args:
            features_df: Member features DataFrame
            
        Returns:
            DataFrame with member_id and score columns for each offer
        """
        scores_df = features_df[["member_id"]].copy()
        
        for offer in self.catalog.offers:
            offer_id = offer["offer_id"]
            base_score = offer["base_score"]
            rules = offer["targeting_rules"]
            
            # Calculate score for each member
            scores = np.full(len(features_df), base_score, dtype=float)
            
            for rule_name, rule_config in rules.items():
                scores += self._apply_rule(features_df, rule_name, rule_config)
            
            # Add noise for realism (simulate unknown factors)
            noise = np.random.normal(0, 5, len(features_df))
            scores += noise
            
            # Normalize to 0-100 range
            scores = np.clip(scores, 0, 100)
            
            scores_df[f"score_{offer_id}"] = scores
            
        return scores_df
    
    def _apply_rule(
        self, 
        df: pd.DataFrame, 
        rule_name: str, 
        rule_config: Any
    ) -> np.ndarray:
        """Apply a single targeting rule"""
        
        boost = np.zeros(len(df))
        
        # Extract feature name from rule name
        feature_name = rule_name.replace("_boost", "")
        
        # Handle different feature name patterns
        possible_names = [
            feature_name,
            f"{feature_name}_encoded",
            f"has_{feature_name}",
            feature_name.replace("_", "_"),
        ]
        
        # Find matching column
        col_name = None
        for name in possible_names:
            if name in df.columns:
                col_name = name
                break
        
        if col_name is None:
            return boost
        
        if isinstance(rule_config, dict):
            # Threshold-based rule
            if "min" in rule_config:
                mask = df[col_name] >= rule_config["min"]
                boost[mask] = rule_config["boost"]
            elif "max" in rule_config:
                mask = df[col_name] <= rule_config["max"]
                boost[mask] = rule_config["boost"]
        else:
            # Binary flag rule
            boost = df[col_name].fillna(0).values * rule_config
        
        return boost


# COMMAND ----------

class OfferPrioritizationModel:
    """
    ML model for healthcare offer prioritization.
    Uses LightGBM multi-output regressor to predict priority scores.
    """
    
    def __init__(
        self,
        catalog: Optional[OfferCatalog] = None,
        model_params: Optional[Dict] = None
    ):
        self.catalog = catalog or OfferCatalog()
        self.offer_ids = self.catalog.get_offer_ids()
        
        # Default LightGBM parameters
        self.model_params = model_params or {
            "objective": "regression",
            "metric": "rmse",
            "boosting_type": "gbdt",
            "num_leaves": 31,
            "learning_rate": 0.05,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "verbose": -1,
            "n_estimators": 200,
            "random_state": 42
        }
        
        self.model: Optional[MultiOutputRegressor] = None
        self.scaler: Optional[StandardScaler] = None
        self.feature_names: List[str] = []
        self.is_fitted: bool = False
        
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        feature_names: Optional[List[str]] = None,
        scale_features: bool = True
    ) -> Dict[str, float]:
        """
        Train the offer prioritization model
        
        Args:
            X: Feature matrix (without member_id)
            y: Target scores for each offer
            feature_names: Optional list of feature names
            scale_features: Whether to standardize features
            
        Returns:
            Dictionary of training metrics
        """
        self.feature_names = feature_names or list(X.columns)
        
        # Scale features if requested
        if scale_features:
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = X.values if hasattr(X, 'values') else X
        
        # Create base LightGBM model
        base_model = lgb.LGBMRegressor(**self.model_params)
        
        # Wrap in multi-output regressor
        self.model = MultiOutputRegressor(base_model)
        
        # Fit model
        self.model.fit(X_scaled, y.values if hasattr(y, 'values') else y)
        self.is_fitted = True
        
        # Calculate training metrics
        y_pred = self.model.predict(X_scaled)
        metrics = self._calculate_metrics(y.values if hasattr(y, 'values') else y, y_pred)
        
        return metrics
    
    def predict(
        self, 
        X: pd.DataFrame,
        return_top_n: int = 5
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Predict offer priority scores
        
        Args:
            X: Feature matrix
            return_top_n: Number of top offers to return per member
            
        Returns:
            Tuple of (raw_scores, top_offers_df)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Scale features
        if self.scaler:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X.values if hasattr(X, 'values') else X
        
        # Predict scores
        raw_scores = self.model.predict(X_scaled)
        
        # Clip to 0-100 range
        raw_scores = np.clip(raw_scores, 0, 100)
        
        # Create top offers DataFrame
        top_offers = self._get_top_offers(raw_scores, return_top_n)
        
        return raw_scores, top_offers
    
    def _get_top_offers(self, scores: np.ndarray, top_n: int) -> pd.DataFrame:
        """Get top N offers for each member"""
        results = []
        
        for i, member_scores in enumerate(scores):
            # Get sorted indices
            sorted_idx = np.argsort(member_scores)[::-1][:top_n]
            
            for rank, idx in enumerate(sorted_idx, 1):
                offer_id = self.offer_ids[idx]
                offer = self.catalog.get_offer_by_id(offer_id)
                
                results.append({
                    "member_idx": i,
                    "rank": rank,
                    "offer_id": offer_id,
                    "offer_name": offer["name"] if offer else "Unknown",
                    "category": offer["category"] if offer else "Unknown",
                    "priority_score": round(member_scores[idx], 2)
                })
        
        return pd.DataFrame(results)
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate model performance metrics"""
        return {
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "r2": float(r2_score(y_true, y_pred)),
            "avg_rmse_per_offer": float(np.mean([
                np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i]))
                for i in range(y_true.shape[1])
            ]))
        }
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance from the model"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        # Average importance across all output models
        importances = np.zeros(len(self.feature_names))
        
        for estimator in self.model.estimators_:
            importances += estimator.feature_importances_
        
        importances /= len(self.model.estimators_)
        
        importance_df = pd.DataFrame({
            "feature": self.feature_names,
            "importance": importances
        }).sort_values("importance", ascending=False)
        
        return importance_df
    
    def save(self, path: str):
        """Save model artifacts"""
        artifacts = {
            "model": self.model,
            "scaler": self.scaler,
            "feature_names": self.feature_names,
            "offer_ids": self.offer_ids,
            "model_params": self.model_params,
            "is_fitted": self.is_fitted
        }
        joblib.dump(artifacts, path)
    
    @classmethod
    def load(cls, path: str) -> "OfferPrioritizationModel":
        """Load model from artifacts"""
        artifacts = joblib.load(path)
        
        model = cls(model_params=artifacts["model_params"])
        model.model = artifacts["model"]
        model.scaler = artifacts["scaler"]
        model.feature_names = artifacts["feature_names"]
        model.offer_ids = artifacts["offer_ids"]
        model.is_fitted = artifacts["is_fitted"]
        
        return model


# COMMAND ----------

def create_training_data(
    features_df: pd.DataFrame,
    catalog: Optional[OfferCatalog] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create training data with target scores
    
    Args:
        features_df: Member features with member_id
        catalog: Offer catalog (optional)
        
    Returns:
        Tuple of (X, y) DataFrames
    """
    scorer = RuleBasedScorer(catalog)
    scores_df = scorer.generate_target_scores(features_df)
    
    # Prepare X (features) and y (targets)
    X = features_df.drop(columns=["member_id"])
    y = scores_df.drop(columns=["member_id"])
    
    return X, y


# COMMAND ----------

# MAGIC %md
# MAGIC ## Sample Usage
# MAGIC ```python
# MAGIC # Create training data
# MAGIC X, y = create_training_data(features_df)
# MAGIC 
# MAGIC # Train model
# MAGIC model = OfferPrioritizationModel()
# MAGIC metrics = model.fit(X, y, feature_names=list(X.columns))
# MAGIC 
# MAGIC # Get predictions
# MAGIC scores, top_offers = model.predict(X, return_top_n=5)
# MAGIC 
# MAGIC # Feature importance
# MAGIC importance = model.get_feature_importance()
# MAGIC ```

