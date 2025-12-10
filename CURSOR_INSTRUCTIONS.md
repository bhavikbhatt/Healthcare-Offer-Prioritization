# Cursor Instructions: Healthcare Offer Prioritization System

> **Purpose**: This file contains all the instructions, patterns, and lessons learned for building this project. Provide this file to Cursor (or any AI coding assistant) when you want to build a similar application or extend this one.

---

## Project Overview

Build an end-to-end ML system on Databricks that personalizes healthcare offers for insurance members. The system should:

1. Generate synthetic healthcare data (members, claims, benefits, engagement)
2. Engineer 80+ ML features from raw data
3. Train a multi-output LightGBM model to predict priority scores for 16 offers
4. Generate personalized recommendations with SHAP explainability
5. Use an LLM to generate natural language reasoning for each recommendation

---

## Architecture Requirements

```
Raw Data → Feature Engineering → ML Model → Recommendations → SHAP + LLM Explanations
```

### Tech Stack
- **Platform**: Databricks (notebooks as .py files with `# COMMAND ----------` separators)
- **ML Framework**: LightGBM with sklearn MultiOutputRegressor
- **Experiment Tracking**: MLflow
- **Model Registry**: Databricks Unity Catalog
- **Explainability**: SHAP TreeExplainer
- **LLM**: Databricks Foundation Model API (OpenAI-compatible)

---

## Project Structure to Create

```
project_name/
├── config/
│   ├── __init__.py
│   └── config.py              # Dataclass configs for Databricks, Data, Offers, Model
├── data/
│   ├── __init__.py
│   └── generate_synthetic_data.py  # Functions to generate realistic healthcare data
├── features/
│   ├── __init__.py
│   └── feature_engineering.py      # FeatureEngineer class
├── models/
│   ├── __init__.py
│   └── offer_model.py              # OfferCatalog, RuleBasedScorer, OfferPrioritizationModel
├── notebooks/
│   ├── __init__.py
│   ├── 01_data_exploration.py
│   ├── 02_feature_engineering.py
│   ├── 03_model_training.py
│   └── 04_model_inference.py
├── utils/
│   ├── __init__.py
│   └── helpers.py
├── requirements.txt
├── setup.py
└── README.md
```

---

## Step-by-Step Build Instructions

### Step 1: Create Data Generation Module

**File**: `data/generate_synthetic_data.py`

Create functions to generate:
- `generate_members(n_members)` - Demographics with age, gender, region, plan_type, tenure, income, risk_score
- `generate_claims(members_df)` - Claims with types (primary_care, specialist, ER, etc.), amounts, diagnoses
- `generate_benefits(members_df)` - Benefits utilization by type (medical, pharmacy, dental, vision, mental_health)
- `generate_engagement(members_df)` - Engagement history by channel (email, app, portal, phone)
- `generate_all_data(n_members, seed)` - Main entry point returning all 4 DataFrames

**Key patterns**:
- Use numpy random with seed for reproducibility
- Create realistic distributions (e.g., claims amounts follow log-normal)
- Add correlations (e.g., older members have more claims, chronic conditions)

---

### Step 2: Create Feature Engineering Module

**File**: `features/feature_engineering.py`

Create `FeatureEngineer` class with methods:
- `create_features()` - Main orchestrator
- `_create_demographic_features()` - Age groups, tenure groups, encoded categoricals
- `_create_claims_features()` - Aggregations, time windows, claim types
- `_create_diagnosis_features()` - Condition flags (has_diabetes, has_cardiovascular, etc.)
- `_create_benefits_features()` - Utilization rates by benefit type
- `_create_engagement_features()` - Channel preferences, response rates
- `_create_temporal_features()` - Seasonal flags, days until year end

**⚠️ CRITICAL LESSON #1: Boolean vs Series with .astype()**

```python
# ❌ WRONG - This creates a Python bool, not a Series
df["is_flu_season"] = (now.month in [10, 11, 12, 1, 2, 3]).astype(int)

# ✅ CORRECT - Use int() for scalar booleans
df["is_flu_season"] = int(now.month in [10, 11, 12, 1, 2, 3])

# ✅ CORRECT - .astype() works on Series comparisons
df["is_senior"] = (members_df["age"] >= 65).astype(int)
```

**⚠️ CRITICAL LESSON #2: Encode categorical columns from the correct DataFrame**

```python
# ❌ WRONG - age_group is in df, not members_df, so this never executes
df["age_group"] = pd.cut(members_df["age"], bins=..., labels=["18-25", "26-35", ...]).astype(str)
for col in ["gender", "region", "age_group"]:
    if col in members_df.columns:  # age_group is NOT in members_df!
        df[f"{col}_encoded"] = encode(members_df[col])

# ✅ CORRECT - Check the right DataFrame and drop string columns
df["age_group"] = pd.cut(members_df["age"], ...).astype(str)
df["age_group_encoded"] = encode(df["age_group"])  # Encode from df
df = df.drop(columns=["age_group"])  # Remove string column before ML
```

---

### Step 3: Create Model Module

**File**: `models/offer_model.py`

Create three classes:

1. **OfferCatalog** - Dataclass with list of offer dictionaries
   - Each offer has: offer_id, name, category, base_score, targeting_rules
   - Include methods: `get_offer_ids()`, `get_offer_by_id()`

2. **RuleBasedScorer** - Generates training labels from business rules
   - Method: `generate_target_scores(features_df)` returns scores 0-100 per offer
   - Apply rule boosts based on member features

3. **OfferPrioritizationModel** - Wrapper around MultiOutputRegressor
   - `fit(X, y)` - Train with StandardScaler + LightGBM
   - `predict(X, return_top_n=5)` - Return raw scores + top offers DataFrame
   - `get_feature_importance()` - Average importance across all estimators
   - `save(path)` / `load(path)` - Joblib serialization

---

### Step 4: Create Training Notebook

**File**: `notebooks/03_model_training.py`

Key sections:
1. Load/generate data
2. Create features using FeatureEngineer
3. Create training data with RuleBasedScorer
4. Train model with MLflow tracking
5. Log metrics, artifacts, feature importance plot
6. Register model in Unity Catalog

**⚠️ CRITICAL LESSON #3: Variable shadowing in notebooks**

```python
# ❌ WRONG - Sample predictions cell overwrites raw_scores
sample_X = X_test.head(5)
raw_scores, top_offers = model.predict(sample_X)  # Now raw_scores has 5 rows!

# Later cell tries to use raw_scores for full test set
rmse = mean_squared_error(y_test, raw_scores)  # ERROR: 10000 vs 5 samples

# ✅ CORRECT - Use different variable names
sample_scores, top_offers = model.predict(sample_X)

# Or recompute before RMSE calculation
raw_scores, _ = model.predict(X_test)  # Full test set
```

---

### Step 5: Create Inference Notebook

**File**: `notebooks/04_model_inference.py`

Key sections:
1. Load model from Unity Catalog (or train fresh for demo)
2. Generate recommendations with filtering
3. Compute SHAP explanations per member
4. Generate LLM reasoning for each recommendation
5. Export results

**⚠️ CRITICAL LESSON #4: Unity Catalog doesn't support "latest" version**

```python
# ❌ WRONG - "latest" not supported in Unity Catalog
model_uri = f"models:/{model_name}/latest"

# ✅ CORRECT - Use specific version
model_uri = f"models:/{model_name}/1"

# ✅ CORRECT - Or use alias
model_uri = f"models:/{model_name}@champion"
```

**⚠️ CRITICAL LESSON #5: Loaded sklearn model lacks custom methods**

```python
# ❌ WRONG - MLflow returns raw sklearn model, not your wrapper
sklearn_model = mlflow.sklearn.load_model(model_uri)
sklearn_model.predict(X, return_top_n=5)  # ERROR: unexpected argument

# ✅ CORRECT - Wrap the loaded model back into your custom class
def wrap_loaded_model(sklearn_model, catalog, feature_names, scaler=None):
    wrapper = OfferPrioritizationModel(catalog=catalog)
    wrapper.model = sklearn_model
    wrapper.feature_names = feature_names
    wrapper.scaler = scaler
    wrapper.is_fitted = True
    return wrapper

# Or load the full wrapper from joblib artifact
model = OfferPrioritizationModel.load(joblib_path)
```

**⚠️ CRITICAL LESSON #6: Databricks Foundation Model API URL format**

```python
# ❌ WRONG - Missing https:// prefix
workspace_url = spark.conf.get("spark.databricks.workspaceUrl")
base_url = f"{workspace_url}/serving-endpoints"  # May be missing https://

# ✅ CORRECT - Ensure https:// prefix
workspace_url = spark.conf.get("spark.databricks.workspaceUrl")
if not workspace_url.startswith("https://"):
    workspace_url = f"https://{workspace_url}"
base_url = f"{workspace_url}/serving-endpoints"
```

---

### Step 6: Add SHAP Explainability

```python
import shap

def compute_member_explanations(model, features_df, recommendations_df, top_features=5):
    """Compute per-member feature importance using SHAP."""
    
    # Create explainer for each offer model (estimator in MultiOutputRegressor)
    shap_values_all = []
    for estimator in model.model.estimators_:
        explainer = shap.TreeExplainer(estimator)
        shap_values = explainer.shap_values(X_scaled)
        shap_values_all.append(shap_values)
    
    # For each recommendation, get top contributing features
    for member_id, offer_id in recommendations:
        member_idx = ...  # Find member index
        offer_idx = model.offer_ids.index(offer_id)
        member_shap = shap_values_all[offer_idx][member_idx]
        
        # Get top features by absolute SHAP value
        top_indices = np.argsort(np.abs(member_shap))[::-1][:top_features]
```

---

### Step 7: Add LLM Reasoning Generation

```python
from openai import OpenAI

def get_llm_client():
    """Initialize client for Databricks Foundation Model API."""
    try:
        # Databricks environment
        workspace_url = spark.conf.get("spark.databricks.workspaceUrl")
        if not workspace_url.startswith("https://"):
            workspace_url = f"https://{workspace_url}"
        
        api_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
        
        client = OpenAI(
            api_key=api_token,
            base_url=f"{workspace_url}/serving-endpoints"
        )
        return client, "databricks-meta-llama-3-1-70b-instruct"
    except:
        # Fallback to environment variables
        api_key = os.environ.get("OPENAI_API_KEY")
        client = OpenAI(api_key=api_key)
        return client, "gpt-4o-mini"


def generate_offer_reasoning(member_data, offer_data, client, model):
    """Generate 3-4 sentence explanation for a recommendation."""
    
    prompt = f"""You are a healthcare benefits advisor.
    
    MEMBER: Age {member_data['age']}, Risk Score {member_data['risk_score']}
    OFFER: {offer_data['offer_name']}
    KEY FACTORS: {offer_data['key_factors']}
    
    Write 3-4 sentences explaining why this offer is recommended."""
    
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200
    )
    return response.choices[0].message.content
```

---

## Common Patterns to Follow

### Databricks Notebook Format
```python
# Databricks notebook source
# MAGIC %md
# MAGIC # Title

# COMMAND ----------

# Code cell 1

# COMMAND ----------

# Code cell 2
```

### MLflow Logging Pattern
```python
with mlflow.start_run(run_name="my_run") as run:
    # Log parameters
    mlflow.log_params({"learning_rate": 0.05, "n_estimators": 200})
    
    # Train model
    model.fit(X_train, y_train)
    
    # Log metrics
    mlflow.log_metric("rmse", rmse_value)
    
    # Log artifacts
    mlflow.log_artifact("/tmp/feature_importance.png")
    
    # Log model
    mlflow.sklearn.log_model(model, "model", signature=signature)
```

### Feature Engineering Pattern
```python
class FeatureEngineer:
    def __init__(self, reference_date=None):
        self.reference_date = reference_date or datetime.now()
        self.label_encoders = {}
        self.feature_names = []
    
    def create_features(self, members_df, claims_df, ...):
        # Create feature groups
        demo_features = self._create_demographic_features(members_df)
        claims_features = self._create_claims_features(claims_df)
        ...
        
        # Merge all on member_id
        features_df = demo_features
        for df in [claims_features, ...]:
            features_df = features_df.merge(df, on="member_id", how="left")
        
        # Handle missing values
        features_df = self._handle_missing_values(features_df)
        
        return features_df
```

---

## Testing Checklist

Before running the full pipeline, verify:

- [ ] All categorical columns are encoded (no strings in feature matrix)
- [ ] No NaN values in features (fill with 0, median, or 365 for days_since)
- [ ] Feature count matches between training and inference
- [ ] Model wrapper has all required methods (predict with return_top_n)
- [ ] SHAP explainer uses the same scaled features as model
- [ ] LLM endpoint is accessible and returns valid responses

---

## Dependencies

```
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
lightgbm>=3.3.5
mlflow>=2.8.0
shap>=0.42.0
openai>=1.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
```

---

## Summary of All Lessons Learned

| Issue | Symptom | Solution |
|-------|---------|----------|
| Bool vs Series | `'bool' object has no attribute 'astype'` | Use `int()` for scalar booleans, `.astype(int)` only for Series |
| String columns in ML | `could not convert string to float: '65+'` | Encode categoricals and drop original string columns |
| Variable shadowing | `inconsistent numbers of samples: [10000, 5]` | Use unique variable names; recompute before metrics |
| Unity Catalog versioning | `'get_latest_versions' is unsupported` | Use version number (`/1`) or alias (`@champion`) |
| Loaded model missing methods | `unexpected keyword argument 'return_top_n'` | Wrap sklearn model back into custom class |
| LLM connection error | `Connection error` | Add `https://` prefix to workspace URL |

---

## Extending This Project

To add new features:
1. Add new method `_create_X_features()` in FeatureEngineer
2. Call it in `create_features()` and merge on member_id
3. Re-run notebooks 02, 03, 04

To add new offers:
1. Add offer dict to `OfferCatalog.offers` list
2. Add targeting rules in `RuleBasedScorer._apply_rule()`
3. Re-train model (notebook 03)

To change LLM:
1. Update `LLM_ENDPOINT_NAME` in notebook 04
2. Or set `OPENAI_API_KEY` environment variable for external API

---

*This instruction file was generated from a real development session. All lessons are from actual bugs encountered and fixed.*

