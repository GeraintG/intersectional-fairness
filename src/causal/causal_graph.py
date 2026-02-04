import pandas as pd
from pgmpy.estimators import HillClimbSearch, BicScore
from pgmpy.models import BayesianModel
from pgmpy.inference import VariableElimination
from pgmpy.estimators import BayesianEstimator

class CausalGraph:
    def __init__(self, df: pd.DataFrame, target: str):
        self.df = df.copy()
        self.target = target
        self.model = None
        self.infer = None

    def fit(self):
        hc = HillClimbSearch(self.df)
        best_model = hc.estimate(scoring_method=BicScore(self.df))
        self.model = BayesianModel(best_model.edges())
        self.model.fit(self.df, estimator=BayesianEstimator, prior_type='BDeu')
        self.infer = VariableElimination(self.model)

    def mitigate_edge_relation(self, sensible_feature):
        """
        移除敏感属性到目标变量的边（如果存在）以干预因果偏见路径。
        """
        if self.model is None:
            self.fit()
        for attr in sensible_feature:
            if (attr, self.target) in self.model.edges():
                self.model.remove_edge(attr, self.target)
        self.model.fit(self.df, estimator=BayesianEstimator, prior_type='BDeu')
        self.infer = VariableElimination(self.model)

    def generate_dataset(self, n_samples=1000, methodtype='bayes'):
        """
        生成因果干预后的公平数据。
        """
        sampled = self.model.simulate(n_samples=n_samples)
        return sampled
