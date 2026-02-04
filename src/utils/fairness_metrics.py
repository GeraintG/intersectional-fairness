import pandas as pd
import numpy as np
from typing import List, Optional


def compute_df_epsilon(
    df: pd.DataFrame,
    sensitive_col: str,
    label_col: str,
    prediction_col: Optional[str] = None,
    epsilon: float = 1e-6
) -> float:
    """
    基于 Differential Fairness 定义计算最大 log 概率差 (ε)。
    """
    values = df[sensitive_col].dropna().unique()
    group_probs = []
    for v in values:
        group_df = df[df[sensitive_col] == v]
        if len(group_df) == 0:
            continue
        if prediction_col:
            p = group_df[prediction_col].mean()
        else:
            p = group_df[label_col].mean()
        p = np.clip(p, epsilon, 1 - epsilon)
        group_probs.append(np.log(p))

    if len(group_probs) < 2:
        return 0.0

    gaps = [abs(p1 - p2) for i, p1 in enumerate(group_probs) for j, p2 in enumerate(group_probs) if i < j]
    return max(gaps)


def compute_statistical_parity_difference(
    df: pd.DataFrame,
    sensitive_col: str,
    label_col: str
) -> float:
    """
    计算 Statistical Parity Difference (SPD):
    P(ŷ=1|s=1) - P(ŷ=1|s=0)
    """
    groups = df[sensitive_col].dropna().unique()
    if len(groups) != 2:
        raise ValueError("SPD 仅支持二分类敏感属性")

    g0, g1 = groups[0], groups[1]
    p0 = df[df[sensitive_col] == g0][label_col].mean()
    p1 = df[df[sensitive_col] == g1][label_col].mean()
    return p1 - p0


def compute_group_metrics_all(
    df: pd.DataFrame,
    sensitive_cols: List[str],
    label_col: str = "label",
    prediction_col: Optional[str] = None
):
    """
    批量计算每个敏感属性或组合的 DF ε 值与 SPD 值。
    """
    for col in sensitive_cols:
        epsilon = compute_df_epsilon(df, col, label_col, prediction_col)
        try:
            spd = compute_statistical_parity_difference(df, col, prediction_col or label_col)
        except ValueError:
            spd = None
        print(f"[{col}]  DF ε: {epsilon:.4f} | SPD: {spd if spd is not None else 'n/a'}")
