import yaml
import pandas as pd
from src.utils.loader import load_data, preprocess_data, split_data
from src.models.tabular_baseline import TabularBaseline
from src.utils.logger import setup_logger

def run_baseline(config_file):
    """
    Run the baseline experiment.
    
    Args:
        config_file: Path to the configuration file
        
    Returns:
        None
    """
    # Load configuration
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    
    # Set up logger
    logger = setup_logger(config["general"]["output_dir"])
    
    # Load and preprocess data
    df = load_data(config["data"]["raw_dir"], "train.csv")
    X, y, sensitive_df = preprocess_data(df, config["fairness"]["sensitive_attributes"])
    X_train, X_val, y_train, y_val = split_data(X, y, test_size=0.2, random_state=config["general"]["random_seed"])
    
    # Initialize and train the baseline model
    baseline_model = TabularBaseline(model_type=config["model"]["type"], random_state=config["general"]["random_seed"])
    baseline_model.fit(X_train, y_train)
    
    # Evaluate the model
    train_accuracy = baseline_model.score(X_train, y_train)
    val_accuracy = baseline_model.score(X_val, y_val)
    
    logger.info(f"Train accuracy: {train_accuracy:.4f}")
    logger.info(f"Validation accuracy: {val_accuracy:.4f}")
    
    # Save the model
    # (Add code to save the model if needed)
    
if __name__ == "__main__":
    config_file = "config/default.yaml"
    run_baseline(config_file)