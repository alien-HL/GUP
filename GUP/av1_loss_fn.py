from typing import Any, Dict, List, Tuple, Union
import os
import numpy as np
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.utils import gpu, to_long


class LossFunc(nn.Module):
    def __init__(self, config, device):
        super(LossFunc, self).__init__()
        self.config = config
        self.device = device
        self.reg_loss = nn.SmoothL1Loss(reduction="sum")

    def forward(self, out_sc, data):
        # （B, N，6，30，2）
        # out_sc = [res_cls_final, res_reg_final, res_reg_fe]
        loss_out = self.pred_loss(
                                  # 网络模型输出：[概率值，最终预测，粗预测]
                                  out_sc,
                                  # 二次转系各车辆坐标（未来30步分割），未来填充步，一次转系各车辆坐标（未来30步分割）
                                  gpu(data["TRAJS_FUT"], self.device),
                                  to_long(gpu(data["PAD_FUT"], self.device)),
                                  gpu(data["TRAJS_FUT_ORI"], self.device),
                                  )
        # 三类损失求和，回归使用-smoothing L1 loss，分类使用-maximum marginal loss，得到总损失
        loss_out["loss"] = (loss_out["cls_loss"] + loss_out["reg_loss"] + loss_out["reg_loss_final"])
        return loss_out

    def pred_loss(self, out_sc: Dict[str, List[torch.Tensor]], gt_preds: List[torch.Tensor], pad_flags: List[torch.Tensor], gt_preds_sc: List[torch.Tensor]):
        #
        cls, reg, reg_final = map(lambda x: torch.cat(x, 0), out_sc[:3])
        gt_preds = torch.cat(gt_preds, 0)
        has_preds = torch.cat(pad_flags, 0).bool()

        #  初始化损失函数字典
        loss_out = dict()
        # 6
        num_modes = self.config["g_num_modes"]
        num_preds = 30

        mask, last_idcs = self.create_mask(has_preds, num_preds)
        # 将无效步填充---gt_preds(N, 30, 2)
        cls, reg, reg_final, gt_preds, has_preds, last_idcs = map(lambda x: x[mask], [cls, reg, reg_final, gt_preds, has_preds, last_idcs])

        # 拿出最后两列坐标，与真实标签做差---得到多模态轨迹差值，最好预测模态距离，最好预测模态索引
        dist, min_dist, min_idcs = self.get_min_distance_indices(reg[..., 0:2].clone(), gt_preds, last_idcs, num_modes)
        # 计算分类损失，粗预测回归损失，最终回归损失
        cls_loss = self.calculate_classification_loss(cls, min_idcs, mask, dist, min_dist)
        reg_loss = self.calculate_regression_loss(reg, min_idcs, gt_preds, has_preds)
        reg_loss_final = self.calculate_regression_loss(reg_final[..., 0:2].clone(), min_idcs, gt_preds, has_preds)
        angle_loss = self.get_angle_diff(reg, min_idcs, gt_preds, has_preds)

        # cls_coef = 0.1, reg_coef = 0.7, reg_coef_final = 0.2
        loss_out["cls_loss"] = self.config["cls_coef"] * cls_loss
        loss_out["reg_loss"] = self.config["reg_coef"] * reg_loss
        loss_out["reg_loss_final"] = self.config["reg_coef_final"] * reg_loss_final
        loss_out["angle_loss"] = 0.5 * angle_loss

        return loss_out

    def create_mask(self, has_preds, num_preds):
        last = has_preds.float() + 0.1 * torch.arange(num_preds).float().to(self.device) / float(num_preds)
        max_last, last_idcs = last.max(1)
        mask = max_last > 1.0
        return mask, last_idcs

    def get_min_distance_indices(self, reg, gt_preds, last_idcs, num_modes):
        row_idcs = torch.arange(len(last_idcs)).long().to(self.device)
        # 计算多模态轨迹与真实标签的差值
        dist = [torch.sqrt(((reg[row_idcs, j, last_idcs] - gt_preds[row_idcs, last_idcs]) ** 2).sum(1)) for j in range(num_modes)]
        dist = torch.stack(dist, dim=1)
        min_dist, min_idcs = dist.min(1)
        # 多模态轨迹距离，最好预测距离，最好预测模态索引
        return dist, min_dist, min_idcs
    
    def calculate_classification_loss(self, cls, min_idcs, mask, dist, min_dist):
        row_idcs = torch.arange(len(min_idcs)).long().to(self.device)
        mgn = cls[row_idcs, min_idcs].unsqueeze(1) - cls
        # cls_th = 2, cls_ignore = 0.2, mgn = 0.2
        mask0 = (min_dist < self.config["cls_th"]).view(-1, 1)
        mask1 = dist - min_dist.view(-1, 1) > self.config["cls_ignore"]
        mgn = mgn[mask0 * mask1]
        mask = mgn < self.config["mgn"]
        num_cls = mask.sum().item()
        cls_loss = (self.config["mgn"] * mask.sum() - mgn[mask].sum()) / (num_cls + 1e-10)
        return cls_loss

    def calculate_regression_loss(self, reg, min_idcs, gt_preds, has_preds):
        # reg---(N, 6, 30, 2)
        row_idcs = torch.arange(len(min_idcs)).long().to(self.device)
        # 最好模态---(N, 30, 2)
        reg = reg[row_idcs, min_idcs]
        # has_preds---(N, 30)
        num_reg = has_preds.sum().item()
        reg_loss = self.reg_loss(reg[has_preds], gt_preds[has_preds]) / (num_reg + 1e-10)
        return reg_loss

    # LAformer中使用的余弦相似度误差
    def get_angle_diff(self, reg, min_idcs, gt_preds, has_preds):
        # 原始模型推理输出 reg---(N, 6, 30, 2)
        # 最好模态索引
        row_idcs = torch.arange(len(min_idcs)).long().to(self.device)
        # 拿出最好模态预测结果---(N, 30, 2)
        reg = reg[row_idcs, min_idcs]
        # has_preds---(N, 30)

        # 计算向量：地面真值与最初点向量: gt_preds[row_idcs]---(85, 30, 2), gt_preds[:, 0, :].unsqueeze---(85, 1, 2)
        gt_traj_angle = gt_preds[row_idcs] - gt_preds[:, 0, :].unsqueeze(1)
        # 计算各预测与最初点向量---（N，30，2）
        pred_traj_angle = reg - gt_preds[:, 0, :].unsqueeze(1)

        # 无效值遮盖操作
        gt_traj_angle = gt_traj_angle[has_preds]
        pred_traj_angle = pred_traj_angle[has_preds]

        # 计算向量夹角
        angle_label = torch.atan2(gt_traj_angle[..., 1], gt_traj_angle[..., 0]).to(torch.float32)
        angle_pred = torch.atan2(pred_traj_angle[..., 1], pred_traj_angle[..., 0]).to(torch.float32)

        # 计算向量差值
        angle_diff = angle_label - angle_pred
        angle_loss = -1 * torch.cos(angle_diff).mean(dim=-1)
        return angle_loss


