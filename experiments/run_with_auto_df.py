import os
import yaml
import pandas as pd
from src.detection.auto_sensitive import detect_sensitive_attributes
from src.models.mlp import MLP
from src.df_training.trainer import Trainer
from src.causal.causal_graph import CausalGraph
from src.utils.logger import setup_logger

# ==== 1. Load config ====
config_path = os.path.join("config", "default.yaml")
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

logger = setup_logger(config["general"]["output_dir"])

# ==== 2. Load and preprocess dataset ====
columns = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"
]
data_path = os.path.join("data", "raw", config["data"]["filename"])
df = pd.read_csv(data_path, header=None, names=columns, na_values=" ?", skipinitialspace=True)
df.dropna(inplace=True)
df["label"] = df["income"].apply(lambda x: 1 if ">50K" in x else 0)

# ==== 3. Auto-detect sensitive attributes ====
candidate_columns = config["fairness"]["candidate_attributes"]
sensitive_results = detect_sensitive_attributes(
    df,
    candidate_columns,
    label_column="label",
    threshold=config["fairness"].get("threshold", 0.1),
    top_k=config["fairness"].get("top_k", 1)
)
sensitive_attrs = [attr for attr, score in sensitive_results]
logger.info(f"Using sensitive attributes: {sensitive_attrs}")

# ==== 4. Apply causal mitigation ====
causal = CausalGraph(df, target="label")
causal.mitigate_edge_relation(sensible_feature=sensitive_attrs)
df_fair = causal.generate_dataset(n_samples=len(df))

# ==== 5. Select features and prepare data ====
exclude_cols = ["income", "label"] + candidate_columns
feature_cols = [c for c in df.columns if c not in exclude_cols]
label_col = "label"

from sklearn.model_selection import train_test_split
X = df_fair[feature_cols]
y = df_fair[label_col]
# Encode combined sensitive attribute as hash (intersectional fairness support)
s_combined = df[sensitive_attrs].astype(str).agg("-".join, axis=1).astype("category").cat.codes

X_train, X_val, y_train, y_val, s_train, s_val = train_test_split(X, y, s_combined, test_size=0.2, random_state=config["general"].get("random_seed", 42))

# ==== 6. Train model ====
model = MLP(input_dim=len(feature_cols))
trainer = Trainer(model, config)
trainer.train(X_train, y_train, s_train, X_val, y_val, s_val)
