import pandas as pd
import numpy as np
from typing import List, Tuple


def compute_group_probability_gap(
    df: pd.DataFrame,
    sensitive_column: str,
    label_column: str,
    prediction_column: str = None,
    epsilon: float = 1e-6
) -> float:
    """
    计算某一敏感属性在不同群体下预测（或标签）概率的 log 比差值。
    """
    values = df[sensitive_column].dropna().unique()
    group_probs = []
    for v in values:
        group_df = df[df[sensitive_column] == v]
        if len(group_df) == 0:
            continue
        if prediction_column:
            p = group_df[prediction_column].mean()
        else:
            p = group_df[label_column].mean()
        p = np.clip(p, epsilon, 1 - epsilon)
        group_probs.append(np.log(p))

    gaps = [abs(p1 - p2) for i, p1 in enumerate(group_probs) for j, p2 in enumerate(group_probs) if i < j]
    return max(gaps) if gaps else 0.0


def detect_sensitive_attributes(
    df: pd.DataFrame,
    candidate_columns: List[str],
    label_column: str,
    prediction_column: str = None,
    threshold: float = 0.2,
    top_k: int = 3,
    support_intersection: bool = True
) -> List[Tuple[str, float]]:
    """
    自动检测敏感属性，返回按公平性分数排序的列表，可选择 top-k 或设置阈值。
    若启用交叉识别，则会评估多属性组合（如 sex+race）。
    """
    scores = []
    columns_to_check = candidate_columns.copy()

    if support_intersection and len(candidate_columns) >= 2:
        for i in range(len(candidate_columns)):
            for j in range(i + 1, len(candidate_columns)):
                combined_col = f"{candidate_columns[i]}+{candidate_columns[j]}"
                df[combined_col] = df[candidate_columns[i]].astype(str) + "-" + df[candidate_columns[j]].astype(str)
                columns_to_check.append(combined_col)

    for col in columns_to_check:
        score = compute_group_probability_gap(
            df, col, label_column, prediction_column
        )
        scores.append((col, score))

    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    filtered = [(col, s) for col, s in scores if s >= threshold]
    return filtered[:top_k] if top_k else filtered
