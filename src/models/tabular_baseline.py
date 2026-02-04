from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

class TabularBaseline:
    def __init__(self, model_type="logistic_regression", **kwargs):
        """
        Initialize a tabular baseline model.
        
        Args:
            model_type: Type of model ("logistic_regression", "random_forest", "xgboost")
            **kwargs: Additional keyword arguments for the model
        """
        self.model_type = model_type
        self.model = self._initialize_model(model_type, **kwargs)
        
    def _initialize_model(self, model_type, **kwargs):
        """
        Initialize the specified model.
        
        Args:
            model_type: Type of model
            **kwargs: Additional keyword arguments for the model
            
        Returns:
            Initialized model
        """
        if model_type == "logistic_regression":
            return LogisticRegression(**kwargs)
        elif model_type == "random_forest":
            return RandomForestClassifier(**kwargs)
        elif model_type == "xgboost":
            return XGBClassifier(**kwargs)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
    def fit(self, X, y):
        """
        Fit the model to the data.
        
        Args:
            X: Features
            y: Labels
            
        Returns:
            None
        """
        self.model.fit(X, y)
        
    def predict(self, X):
        """
        Predict labels for the given features.
        
        Args:
            X: Features
            
        Returns:
            Predicted labels
        """
        return self.model.predict(X)