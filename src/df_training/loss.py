import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from typing import List


def compute_df_loss(
    predictions: torch.Tensor,
    sensitive_values: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float = 1e-6
) -> torch.Tensor:
    """
    基于 DF 概念计算 log 概率比作为公平性正则项。
    输入：
        predictions: 模型预测概率, (batch,)
        sensitive_values: 整数编码的敏感属性组值, (batch,)
        labels: 实际标签值 (0/1), (batch,)
    输出：
        一个标量 loss 表示最大组间 log 比例偏差
    """
    group_ids = sensitive_values.unique()
    group_probs = []
    for gid in group_ids:
        idx = (sensitive_values == gid)
        if idx.sum() < 2:
            continue
        group_pred = predictions[idx]
        group_avg = torch.clamp(group_pred.mean(), epsilon, 1 - epsilon)
        group_log = torch.log(group_avg)
        group_probs.append(group_log)

    if len(group_probs) < 2:
        return torch.tensor(0.0, requires_grad=True)

    diffs = [torch.abs(p1 - p2) for i, p1 in enumerate(group_probs) for j, p2 in enumerate(group_probs) if i < j]
    return torch.stack(diffs).max()


class FairLoss(nn.Module):
    def __init__(self, lambda_df: float = 0.5):
        super().__init__()
        self.lambda_df = lambda_df
        self.bce = nn.BCELoss()

    def forward(self, predictions, labels, sensitive_values):
        loss_task = self.bce(predictions, labels.float())
        loss_df = compute_df_loss(predictions, sensitive_values, labels)
        return loss_task + self.lambda_df * loss_df
