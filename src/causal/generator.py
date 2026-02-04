import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from causalml.inference.meta import LRSRegressor
from causalml.inference.tree import UpliftTreeClassifier, UpliftRandomForestClassifier

class CausalDataGenerator:
    def __init__(self, treatment_col, outcome_col, sensitive_cols):
        """
        Initialize a causal data generator.
        
        Args:
            treatment_col: Name of the treatment column
            outcome_col: Name of the outcome column
            sensitive_cols: List of sensitive attribute column names
        """
        self.treatment_col = treatment_col
        self.outcome_col = outcome_col
        self.sensitive_cols = sensitive_cols
        
    def generate_data(self, df, n_samples=1000):
        """
        Generate synthetic data using a causal model.
        
        Args:
            df: Original data DataFrame
            n_samples: Number of samples to generate
            
        Returns:
            Synthetic data DataFrame
        """
        # Preprocess data
        X = df.drop(columns=[self.treatment_col, self.outcome_col] + self.sensitive_cols)
        y = df[self.outcome_col]
        treatment = df[self.treatment_col]
        sensitive_df = df[self.sensitive_cols]
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train a causal model
        lr = LRSRegressor()
        lr.fit(X_scaled, treatment, y)
        
        # Generate synthetic data
        synthetic_X, synthetic_treatment, synthetic_y = lr.generate(X_scaled, treatment, y, n_samples=n_samples)
        
        # Inverse transform features
        synthetic_X = scaler.inverse_transform(synthetic_X)
        
        # Create synthetic DataFrame
        synthetic_df = pd.DataFrame(synthetic_X, columns=X.columns)
        synthetic_df[self.treatment_col] = synthetic_treatment
        synthetic_df[self.outcome_col] = synthetic_y
        synthetic_df = pd.concat([synthetic_df, sensitive_df], axis=1)
        
        return synthetic_df