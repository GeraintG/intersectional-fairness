import pandas as pd
import os

def load_data(data_dir, file_name):
    """
    Load data from a CSV file.
    
    Args:
        data_dir: Directory containing the data file
        file_name: Name of the data file
        
    Returns:
        Pandas DataFrame containing the data
    """
    file_path = os.path.join(data_dir, file_name)
    return pd.read_csv(file_path)

def preprocess_data(df, sensitive_attributes):
    """
    Preprocess data by encoding categorical variables and splitting into features and labels.
    
    Args:
        df: Pandas DataFrame containing the data
        sensitive_attributes: List of sensitive attribute names
        
    Returns:
        X: Features (Pandas DataFrame)
        y: Labels (Pandas Series)
        sensitive_df: Sensitive attributes (Pandas DataFrame)
    """
    # Separate features, labels, and sensitive attributes
    X = df.drop(columns=["label"] + sensitive_attributes)
    y = df["label"]
    sensitive_df = df[sensitive_attributes]
    
    # Encode categorical variables
    X = pd.get_dummies(X)
    
    return X, y, sensitive_df

def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split data into training and testing sets.
    
    Args:
        X: Features (Pandas DataFrame)
        y: Labels (Pandas Series)
        test_size: Proportion of the dataset to include in the test split
        random_state: Random seed for reproducibility
        
    Returns:
        X_train: Training features
        X_test: Testing features
        y_train: Training labels
        y_test: Testing labels
    """
    from sklearn.model_selection import train_test_split
    return train_test_split(X, y, test_size=test_size, random_state=random_state)