#!/usr/bin/env python
"""
Healthcare Offer Prioritization Demo
====================================

This script runs the complete demo workflow:
1. Generate synthetic member data
2. Create ML features
3. Train the offer prioritization model
4. Generate recommendations
5. Display results

Usage:
    python run_demo.py [--members N] [--verbose]
"""

import argparse
import sys
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')


def run_demo(n_members: int = 10000, verbose: bool = True):
    """
    Run the complete offer prioritization demo
    
    Args:
        n_members: Number of members to generate
        verbose: Print detailed output
    """
    print("\n" + "=" * 70)
    print("🏥 HEALTHCARE OFFER PRIORITIZATION DEMO")
    print("=" * 70)
    print(f"\n📅 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"👥 Members: {n_members:,}")
    
    # =========================================================================
    # Step 1: Generate Synthetic Data
    # =========================================================================
    print("\n" + "-" * 70)
    print("📊 STEP 1: Generating Synthetic Data")
    print("-" * 70)
    
    from data.generate_synthetic_data import generate_all_data
    
    members_df, claims_df, benefits_df, engagement_df = generate_all_data(
        n_members=n_members,
        seed=42
    )
    
    if verbose:
        print(f"\n   Members: {len(members_df):,}")
        print(f"   Claims: {len(claims_df):,}")
        print(f"   Benefits: {len(benefits_df):,}")
        print(f"   Engagements: {len(engagement_df):,}")
    
    # =========================================================================
    # Step 2: Feature Engineering
    # =========================================================================
    print("\n" + "-" * 70)
    print("🔧 STEP 2: Creating Features")
    print("-" * 70)
    
    from features.feature_engineering import create_member_features
    
    features_df, feature_engineer = create_member_features(
        members_df=members_df,
        claims_df=claims_df,
        benefits_df=benefits_df,
        engagement_df=engagement_df
    )
    
    if verbose:
        print(f"\n   Features created: {len(feature_engineer.feature_names)}")
        print(f"   Feature groups:")
        for group, features in feature_engineer.get_feature_importance_groups().items():
            print(f"      {group}: {len(features)}")
    
    # =========================================================================
    # Step 3: Train Model
    # =========================================================================
    print("\n" + "-" * 70)
    print("🤖 STEP 3: Training Model")
    print("-" * 70)
    
    from models.offer_model import (
        OfferPrioritizationModel,
        OfferCatalog,
        create_training_data
    )
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, r2_score
    
    catalog = OfferCatalog()
    X, y = create_training_data(features_df, catalog)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"\n   Training samples: {len(X_train):,}")
    print(f"   Test samples: {len(X_test):,}")
    print(f"   Offers in catalog: {len(catalog.offers)}")
    
    # Train model
    model = OfferPrioritizationModel(catalog=catalog)
    metrics = model.fit(X_train, y_train, feature_names=list(X.columns))
    
    # Evaluate
    raw_scores, _ = model.predict(X_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, raw_scores))
    test_r2 = r2_score(y_test, raw_scores)
    
    print(f"\n   📈 Training Metrics:")
    print(f"      RMSE: {metrics['rmse']:.4f}")
    print(f"      R²: {metrics['r2']:.4f}")
    print(f"\n   📊 Test Metrics:")
    print(f"      RMSE: {test_rmse:.4f}")
    print(f"      R²: {test_r2:.4f}")
    
    # =========================================================================
    # Step 4: Generate Recommendations
    # =========================================================================
    print("\n" + "-" * 70)
    print("🎯 STEP 4: Generating Recommendations")
    print("-" * 70)
    
    # Get predictions for all members
    all_scores, all_recommendations = model.predict(X, return_top_n=5)
    
    # Add member_ids
    all_recommendations['member_id'] = all_recommendations['member_idx'].map(
        lambda x: features_df['member_id'].iloc[x]
    )
    
    print(f"\n   Total recommendations: {len(all_recommendations):,}")
    print(f"   Avg per member: {len(all_recommendations) / len(features_df):.1f}")
    
    # =========================================================================
    # Step 5: Display Results
    # =========================================================================
    print("\n" + "-" * 70)
    print("📋 STEP 5: Results Summary")
    print("-" * 70)
    
    # Top offers
    print("\n   🏆 Top Recommended Offers:")
    offer_counts = all_recommendations['offer_name'].value_counts().head(10)
    for i, (offer, count) in enumerate(offer_counts.items(), 1):
        bar = "█" * int(count / offer_counts.max() * 20)
        print(f"      {i:2d}. {offer[:35]:35s} {bar} {count:,}")
    
    # Category distribution
    print("\n   📊 Recommendations by Category:")
    cat_counts = all_recommendations['category'].value_counts()
    for cat, count in cat_counts.items():
        pct = count / len(all_recommendations) * 100
        print(f"      {cat:20s} {count:6,} ({pct:5.1f}%)")
    
    # Score statistics
    print("\n   📈 Score Distribution:")
    print(f"      Mean: {all_recommendations['priority_score'].mean():.1f}")
    print(f"      Std: {all_recommendations['priority_score'].std():.1f}")
    print(f"      Min: {all_recommendations['priority_score'].min():.1f}")
    print(f"      Max: {all_recommendations['priority_score'].max():.1f}")
    
    # Sample member recommendations
    print("\n   👤 Sample Member Recommendations:")
    sample_members = features_df['member_id'].sample(3, random_state=42).tolist()
    
    for member_id in sample_members:
        member_recs = all_recommendations[
            all_recommendations['member_id'] == member_id
        ].sort_values('rank').head(3)
        
        member_features = features_df[features_df['member_id'] == member_id].iloc[0]
        
        print(f"\n      Member: {member_id}")
        print(f"      Age: {int(member_features['age'])}, "
              f"Risk: {member_features['risk_score']:.1f}, "
              f"Claims: {int(member_features['total_claims_count'])}")
        print(f"      Top Offers:")
        for _, rec in member_recs.iterrows():
            print(f"         {rec['rank']}. {rec['offer_name']} "
                  f"(Score: {rec['priority_score']:.1f})")
    
    # Feature importance
    print("\n   🔍 Top 10 Most Important Features:")
    importance = model.get_feature_importance().head(10)
    for i, row in importance.iterrows():
        bar = "█" * int(row['importance'] / importance['importance'].max() * 20)
        print(f"      {row['feature'][:30]:30s} {bar}")
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("✅ DEMO COMPLETE")
    print("=" * 70)
    print(f"\n📅 Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\n📊 Summary:")
    print(f"   • Processed {n_members:,} members")
    print(f"   • Created {len(feature_engineer.feature_names)} features")
    print(f"   • Trained model with R² = {test_r2:.4f}")
    print(f"   • Generated {len(all_recommendations):,} recommendations")
    print(f"   • Covering {len(catalog.offers)} offer types")
    print("\n💡 Next Steps:")
    print("   • Run notebooks in Databricks for full MLflow integration")
    print("   • Register model in Databricks Model Registry")
    print("   • Set up batch inference pipeline")
    print("   • Connect to marketing platforms for campaign execution")
    print("=" * 70 + "\n")
    
    return {
        "features_df": features_df,
        "model": model,
        "recommendations": all_recommendations,
        "metrics": {"train": metrics, "test": {"rmse": test_rmse, "r2": test_r2}}
    }


def main():
    parser = argparse.ArgumentParser(
        description="Healthcare Offer Prioritization Demo"
    )
    parser.add_argument(
        "--members", "-m",
        type=int,
        default=10000,
        help="Number of members to generate (default: 10000)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        default=True,
        help="Print detailed output"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Minimal output"
    )
    
    args = parser.parse_args()
    
    verbose = not args.quiet
    
    try:
        results = run_demo(n_members=args.members, verbose=verbose)
        return 0
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

