from typing import Any, Dict, List, Tuple, Union, Optional
import time
import math
import numpy as np
from fractions import gcd

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn import MultiheadAttention

from utils.utils import gpu, init_weights


class GUP(nn.Module):
    # Initialization
    def __init__(self, cfg, device):
        super(GUP, self).__init__()
        self.device = device
        
        self.g_num_modes = 6
        self.g_pred_len = 30
        self.g_num_coo = 2
        self.d_embed = 128

        # ------------------------------------------车辆节点编码---------------------------------------------
        # 二次转系下的各车辆编码：参数3，128，3
        #self.actor_net_ac = Actor_Encoder(device=self.device,
        #                          n_in=cfg['in_actor'],
        #                         hidden_size=cfg['d_actor'],
        #                          n_fpn_scale=cfg['n_fpn_scale'])
        # 一次转系下的各车辆编码
        #self.actor_net_sc = Actor_Encoder(device=self.device,
        #                                  n_in=cfg['in_actor'],
        #                                  hidden_size=cfg['d_actor'],
        #                                  n_fpn_scale=cfg['n_fpn_scale'])

        self.actor_lstm = Actor_Encoder_LSTM(device=self.device,
                                             n_in=cfg['in_actor'],
                                             hidden_size=cfg['d_actor'],
                                             n_fpn_scale=cfg['n_fpn_scale'])


        # ------------------------------------------车道节点编码---------------------------------------------
        # 二次转系下的各车道编码：参数：10，128，0.1
        self.lane_net_ac = Map_Encoder(device=self.device,
                                in_size=cfg['in_lane'],
                                hidden_size=cfg['d_lane'],
                                dropout=cfg['dropout'])
        # 一次转系下的各车道编码：参数：8，128，0.1
        self.lane_net_sc = Map_Encoder(device=self.device,
                                in_size=cfg['in_lane_sc'],
                                hidden_size=cfg['d_lane'],
                                dropout=cfg['dropout'])

        # 二次转系的车辆与地图交互（rpe）
        #self.ac_am_fusion = FusionNet(device=self.device,
        #                              config=cfg)

        self.ac_sc_fusion = FusionNet2(device=self.device,
                                       config=cfg)

        #self.interaction_module_sc_am = Interaction_Module_SC_AM(device=self.device,
        #                                                       hidden_size = cfg['d_lane'],
        #                                                       depth = cfg["n_interact"])  # 3

        self.interaction_module_af = Interaction_Module_FE(device=self.device,
                                                           hidden_size = cfg['d_lane'],  # d_lane=128
                                                           depth = cfg["n_interact"])  # n_interact=3

        self.interaction_module_al = Interaction_Module_FE(device=self.device,
                                                           hidden_size=cfg['d_lane'],
                                                           depth=cfg["n_interact"])

        #self.interaction_module_m2m = Interaction_Module_M2M(hidden_size=cfg['d_lane'], # 128
        #                                                     depth=cfg["n_interact"])


        self.trajectory_decoder_fe = Trajectory_Decoder_Future_Enhanced(device=self.device,
                                                                       hidden_size = cfg['d_lane'])

        self.trajectory_decoder_occ = Trajectory_Decoder_OCC(device=self.device,
                                                             hidden_size=cfg['d_lane'])

        # 采用GRU解码器
        #self.GRUdecoder = GRUDecoder(local_channels=self.d_embed*2,
        #                             global_channels=self.d_embed*2,
        #                             future_steps=30,
        #                             num_modes=6,
        #                             uncertain=False)
        # 最终解码器
        #self.trajectory_decoder_final = Trajectory_Decoder_Final(device=self.device,
        #                                                        hidden_size = cfg["d_decoder_F"])  # d_decoder_F = 256

        # 多模态微分解码器
        self.differentiation_decoder = Differentiation_Decoder(device=self.device,
                                                               hidden_size=self.d_embed*2,
                                                               num_modes=self.g_num_modes)  # d_decoder_F = 384

        # 最终输出解码器
        #self.trajectory_decoder_all = Trajectory_Decoder_All(device=self.device,
        #                                                     hidden_size=self.d_embed*4,)  # d_decoder_F = 256


        self.rft_encoder = Reliable_Future_Trajectory_Encoder()
        #self.diff_encoder1 = Reliable_Future_Trajectory_Encoder1()
        #self.diff_encoder2 = Reliable_Future_Trajectory_Encoder2()
        self.occ = Occupancy_Encoder()
        #self.mlp = MLP(self.d_embed+1, self.d_embed)
        self.mlp_b = MLP(12, self.d_embed)
        #self.mlp_c = MLP(self.d_embed * 2, self.d_embed)

        #self.lstm_decoder = MultiModalTrajectoryDecoder(feature_dim=self.d_embed,
        #                                                hidden_dim=self.d_embed,
        #                                                num_heads=4,
        #                                                num_layers=2,
        #                                                num_modes=self.g_num_modes,
        #                                               future_steps=self.g_pred_len,
        #                                                output_dim=2)
        self.multihead_proj = nn.Linear(self.d_embed*2, self.g_num_modes * self.d_embed*2)

        #self.aggr_embed = nn.Sequential(
            # 线性层(128，64)
        #    nn.Linear(self.d_embed * 2, self.d_embed),
            # 层归一化（64）
        #    nn.LayerNorm(self.d_embed),
        #    nn.ReLU(inplace=True),
        #    nn.Linear(self.d_embed, self.d_embed))

        if cfg["init_weights"]:
            self.apply(init_weights)


    def forward(self, data):
        actors_ac, actor_idcs, lanes_ac, lane_idcs, rpe, actors_sc, lanes_sc, rpe_sc = data

        # 批次数
        batch_size = len(actor_idcs)

        # -----------------------------------------------SIMPL----------------------------------------------------------------------------------
        # 网络前需要进行前处理操作
        # ac actors/lanes encoding
        # 二次转系下的车辆特征编码，输入维度（N，3，20）->（N，128）
        actors_ac = self.actor_lstm(actors_ac)
        # 二次转系下的车道特征编码，输入维度（M，10，10）->（M，128）
        lanes_ac = self.lane_net_ac(lanes_ac)
        # ac feature fusion为（SFT）SIMPL特征融合(锚姿态) ——> 返回使用边缘(RPE)多头注意力更新后的actors，lanes特征。SIMPL直接用此步actors进行预测
        # actors_ac, _ , _ = self.ac_am_fusion(actors_ac, actor_idcs, lanes_ac, lane_idcs, rpe)


        # -----------------------------------------------GUP---------------------------------------------------------------------------------
        # sc actors/lanes encoding
        # 一次转系下的车辆特征编码，与85行编码方式相同
        actors_sc = self.actor_lstm(actors_sc)
        # 一次转系下的车辆特征编码，与88行编码方式相同，但输入维度（M，10，8）, 8->128,
        lanes_sc = self.lane_net_sc(lanes_sc)
        actors_ac, lanes_ac, actors_sc, lanes_sc = self.ac_sc_fusion(actors_ac, actors_sc, actor_idcs, lanes_ac, lanes_sc, lane_idcs, rpe, rpe_sc)




        # 在一个批次下，各场景文件的车辆数量和车道数量追加列表
        # 车辆数量
        agent_lengths = []
        # 这里的actor_idcs是一个batch的内容，包括许多场景文件的车辆索引列表
        for actor_id in actor_idcs:
            # 计算单批次中，各场景中的agent数量，追加列表
            agent_lengths.append(actor_id.shape[0] if actor_id is not None else 0)
        # 车道数量
        lane_lengths = []
        for lane_id in lane_idcs:
            # 将单批次中，各场景中的lane数量追加车道列表
            lane_lengths.append(lane_id.shape[0] if lane_id is not None else 0)
        
        # 找出含有最多车辆/车道数的场景，作为标准矩阵初始化，下边进行对齐操作
        max_agent_num = max(agent_lengths)
        max_lane_num = max(lane_lengths)

        # 批次对齐操作：batch_size = len(actor_idcs)，batch_size为场景数，也是批次数，一批有多少文件就是多少场景
        # 初始化特征矩阵：一次转系下的车辆编码，二次转系下的车辆编码，一次转系下的车道编码
        actors_batch_sc = torch.zeros(batch_size, max_agent_num, self.d_embed, device=self.device)
        actors_batch_ac = torch.zeros(batch_size, max_agent_num, self.d_embed, device=self.device)
        lanes_batch_sc = torch.zeros(batch_size, max_lane_num, self.d_embed, device=self.device)
        lanes_batch_ac = torch.zeros(batch_size, max_lane_num, self.d_embed, device=self.device)

        # 将车辆/车道节点赋值到标准化矩阵中
        for i, actor_ids in enumerate(actor_idcs):
            # i为某场景，actor_ids为某场景下的车辆索引列表
            # 该场景下的车辆数量
            num_agents = actor_ids.shape[0]
            # 赋值该场景索引下的一次转系车辆特征编码（未任何交互操作）
            actors_batch_sc[i, :num_agents] = actors_sc[actor_ids[0] : actor_ids[-1] + 1]
            # 赋值该场景索引下的二次转系车辆特征编码（通过rpe更新后的节点编码）
            actors_batch_ac[i, :num_agents] = actors_ac[actor_ids[0] : actor_ids[-1] + 1]

        for i, lane_ids in enumerate(lane_idcs):
            num_lanes = lane_ids.shape[0]
            # 赋值该场景索引下的一次转系车道特征编码（未任何交互操作）
            lanes_batch_sc[i, :num_lanes] = lanes_sc[lane_ids[0] : lane_ids[-1] + 1]
            lanes_batch_ac[i, :num_lanes] = lanes_ac[lane_ids[0] : lane_ids[-1] + 1]
        
        # mask为一次转系下的【A-A，A-L，L-L，L-A】遮蔽矩阵初始化，有效置为1，无效置0
        #masks, _ = get_masks(agent_lengths, lane_lengths, self.device)

        # 一次转系下，车辆/车道特征交互模块---使用多头注意力作4个方向的融合，返回更新后的车辆和车道特征，将车辆特征直接用作未来轨迹预测---（batch, N, d128）
        #agent_states, lane_states = self.interaction_module_sc_am(actors_batch_sc, lanes_batch_sc, masks)

        agent_states = actors_batch_sc

        # ------------------------------异质性解码-reliable future trajectory generate------------------------------------
        # 改动成为向量夹角
        # 将一次转系下，更新后的车辆编码直接作解码，产生未来轨迹，输入维度（B, N, 128）---返回第5步的末端点坐标（B，N，6，2），B为batch_size
        predictions_occ = self.trajectory_decoder_occ(agent_states)

        # ---------------------------------------------------改动部分----------------------------------------------------
        # Occupancy---占用网络
        # 将最终预测轨迹拿出---(二次转系坐标下的车辆编码 )
        # 车道占用标识---(B, N, 1)
        # occ_batch_sc = torch.ones(batch_size, max_lane_num, 1, device=self.device)

        # 编码（B，N，6，2）---（B，N，128）
        position5_encoder = self.mlp_b(predictions_occ.reshape(batch_size, max_agent_num, -1))

        # 车道与轨迹中点坐标计算注意力分数---（B，N，D）
        # 遮蔽矩阵---L2A
        masks_la = get_mask(lane_lengths, agent_lengths, self.device)
        # 返回车道占用注意力权重---（B，N，D）
        lane_batch_occ, _ = self.interaction_module_al(lanes_batch_sc, position5_encoder, masks_la)
        # (B, N，D)
        lane_batch_occ_score = F.softmax(lane_batch_occ, dim=-1)

        # 把车道占用符号与车道占用特征拼接---（B，N，D+D）
        lane_occ_fusion = torch.cat((lane_batch_occ, lane_batch_occ_score), dim=-1)
        # 编码占用车道---(B, N, 128)
        occ_state = self.occ(lane_occ_fusion)

        # 将一次转系的占用车道编码与二次转系下的车辆融合作注意力
        # 遮蔽矩阵---A2L
        masks_al_occ = get_mask(agent_lengths, lane_lengths, self.device)
        # 得到一次转系下，融合了前5步的车道占用情况的全局车辆编码
        actors_occ_sc, _ = self.interaction_module_al(actors_batch_sc, occ_state, masks_al_occ)

        # 粗预测
        predictions_fe = self.trajectory_decoder_fe(actors_occ_sc)
        future_states = self.rft_encoder(predictions_fe)

        # ---------------------------------未来特征融合网络-future feature interaction-------------------------------------
        # 遮蔽矩阵---A2A
        masks_af = get_mask_l2a(agent_lengths, agent_lengths, self.device)
        # 一次转系下的未来轨迹编码与二次转系下的车辆编码作3层注意力，返回更新后的二次转系车辆编码，一次转系未来编码。只需取前一部分车辆编码---(B，N，D)
        actors_batch_af, _ = self.interaction_module_af(actors_batch_ac, future_states, masks_af)

        # 遮蔽矩阵---A2L
        masks_al = get_mask(agent_lengths, lane_lengths, self.device)
        # 第二阶段地图交互，使得车辆在融合未来车车交互后重新关注地图部分，让其回归到正确道路上
        # 返回车道更新后的二次转系车辆编码---(B, N, D)
        actors_batch_al, _ = self.interaction_module_al(actors_batch_af, lane_batch_occ, masks_al)


        # 将融合了一次转系未来编码和一次转系地图编码的二次转系车辆编码，与一次转系下通过四层注意力更新的车辆编码作拼接---(B, N, D+D)
        #agent_fusion = torch.cat((actors_batch_al, actors_batch_ac), dim=2)
        agent_fusion = torch.cat((actors_batch_al, actors_batch_ac), dim=-1)
        # 融合一次转系车道，二次转系更新后的车道编码---（B，N，D+D）
        # lane_fusion = torch.cat((lane_batch_la, lane_states), dim=2)
        # 轨迹最终解码，返回多模态轨迹预测结果（B，N，6，30，2），多模态轨迹概率（B，N，6）
        # predictions_future, _ = self.trajectory_decoder_final(agent_fusion)

        # 将解码维度变为（N，F，D）
        B = agent_fusion.shape[0]
        global_map_embed = self.multihead_proj(agent_fusion).view(B, -1, 6, self.d_embed*2)  # [B, N, F, D]

        predictions_final, logits_final = self.differentiation_decoder(agent_fusion, global_map_embed)

        '''
        

        # ------------------------------------------------改动部分-------------------------------------------------------
        # ProLn实现多模态微分——先解码出多模态轨迹，执行M2M注意力，再对其多模态微分，每个模式复杂解码出一条轨迹，拼接返回预测结果---（B，N，6，30，2）
        predictions_differentiation = self.differentiation_decoder(agent_fusion)
        # 结果分割成关键帧---（B，N，6，5，2），后续解码（B，N，6，25，2）
        predictions1 = predictions_differentiation[:, :, :, :5, :]
        # 对输出进行编码(B, N, 6, 5, 2)---(B, N, 128)
        predictions_differentiation1 = self.diff_encoder1(predictions1)
        # 第三部分地图交互，使用一次转系的占用车道信息
        # 遮蔽矩阵---A2L
        masks_al_diff1 = get_mask(agent_lengths, lane_lengths, self.device)
        # (B, N, 128)
        diff_occ_batch_al1, _ = self.interaction_module_al(predictions_differentiation1, occ_state, masks_al_diff1)

        # 后25步
        predictions2 = predictions_differentiation[:, :, :, 5:, :]
        predictions_differentiation2 = self.diff_encoder2(predictions2)
        # 第三部分地图交互，使用一次转系的占用车道信息
        # 遮蔽矩阵---A2L
        masks_al_diff2 = get_mask(agent_lengths, lane_lengths, self.device)
        # (B, N, 128)
        diff_occ_batch_al2, _ = self.interaction_module_al(predictions_differentiation2, lanes_batch_sc, masks_al_diff2)

        # 二次转系融合未来与车道，二次转系交互初始，二次转系关键帧查询车道占用，二次转系后续解码查询一次车道
        # 拼接（actors_batch_al, actors_batch_ac，diff_occ_batch_al1，diff_occ_batch_al2）--- (B, N, 128*4)
        pred_agent_all = torch.cat((agent_fusion, diff_occ_batch_al1, diff_occ_batch_al2), dim=-1)








        # 轨迹最终解码，返回多模态轨迹预测结果（B，N，6，30，2），多模态轨迹概率（B，N，6）
        predictions_final, logits_final = self.trajectory_decoder_all(pred_agent_all)
        '''

        # ------------------------------------------------预测结果赋值----------------------------------------------------
        # 初始化预测结果矩阵：（N，6，30，2），（N，6，30，2），（N，6）
        # actors_sc: 各场景文件车辆编码（N，128）
        # （N，6，30，2）
        expan_predictions_fe = torch.zeros((actors_sc.shape[0], self.g_num_modes, self.g_pred_len, self.g_num_coo), device=self.device)
        expan_predictions_final = torch.zeros((actors_sc.shape[0], self.g_num_modes, self.g_pred_len, self.g_num_coo), device=self.device)
        # （N，6）
        expan_logits_final = torch.zeros((actors_sc.shape[0], self.g_num_modes), device=self.device)

        # 赋值结果矩阵---将一个批次的所有车辆拿出（N，6，30，2）
        # 将每个场景下的车辆id拿出
        for i, actor_ids in enumerate(actor_idcs):
            # 单场景下的车辆数量
            num_agents = actor_ids.shape[0]
            # 粗预测，最终预测，概率分数
            expan_predictions_fe[actor_ids[0]:actor_ids[-1] + 1] = predictions_fe[i, :num_agents]
            #expan_predictions_fe[actor_ids[0]:actor_ids[-1] + 1] = predictions_final[i, :num_agents]
            expan_predictions_final[actor_ids[0]:actor_ids[-1] + 1] = predictions_final[i, :num_agents]
            expan_logits_final[actor_ids[0]:actor_ids[-1] + 1] = logits_final[i, :num_agents]

        res_reg_fe, res_reg_final , res_cls_final = [], [], []
        # i为批次（文件）数量
        for i in range(len(actor_idcs)):
            # 各场景下的所有车辆id列表
            idcs = actor_idcs[i]
            # 为每个场景文件中的所有车辆追加列表---粗预测，最终预测，概率得分
            res_reg_fe.append(expan_predictions_fe[idcs])
            res_reg_final.append(expan_predictions_final[idcs])
            res_cls_final.append(expan_logits_final[idcs])
        # 得到一个批次里的各文件预测结果---(B, N,...)

        # 拼接---（3，B，N, 6, ）
        out_sc = [res_cls_final, res_reg_final, res_reg_fe]
        # 返回轨迹概率值，最终预测，粗预测
        return out_sc

    def pre_process(self, data):
        '''
            Send to device
            'BATCH_SIZE', 'SEQ_ID', 'CITY_NAME',
            'ORIG', 'ROT',
            'TRAJS_OBS', 'TRAJS_FUT', 'PAD_OBS', 'PAD_FUT', 'TRAJS_CTRS', 'TRAJS_VECS',
            'LANE_GRAPH',
            'RPE',
            'ACTORS', 'ACTOR_IDCS', 'LANES', 'LANE_IDCS','LANES_SC','ACTORS_SC'
        '''

        # 网络输入前处理函数，将其迁移到gpu上
        # 数据加载初始化时候无该关键字，需要在av1.dataset.py执行collate_fn整理函数
        actors = gpu(data['ACTORS'], self.device)
        actors_sc = gpu(data['ACTORS_SC'], self.device)
        actor_idcs = gpu(data['ACTOR_IDCS'], self.device)
        lanes = gpu(data['LANES'], self.device)
        lanes_sc = gpu(data['LANES_SC'], self.device)
        lane_idcs = gpu(data['LANE_IDCS'], self.device)

        # 数据加载初始化时候就已经有该关键字
        # 数据加载初始化时候就已经有该关键字
        rpe = gpu(data['RPE'], self.device)
        rpe_sc = gpu(data['RPE_SC'], self.device)
        # 将数据迁移gpu后，进行训练
        return actors, actor_idcs, lanes, lane_idcs, rpe, actors_sc, lanes_sc, rpe_sc

    def post_process(self, out):
        # 推理结果初始化字典
        post_out = dict()
        # out输出3个部分：概率分布，最终预测，粗预测。只需前两个输出即可
        res_cls = out[0]
        res_reg = out[1]

        # get prediction results for target vehicles only
        # 只拿出最感兴趣车辆的轨迹信息：编历回归结果（所有车辆）三部分（agent，av，others）---> 只拿出agent数据，沿dim=0拼接
        reg = torch.stack([trajs[0] for trajs in res_reg], dim=0)
        cls = torch.stack([probs[0] for probs in res_cls], dim=0)

        post_out['out_sc'] = out
        post_out['traj_pred'] = reg  # batch x n_mod x pred_len x 2
        post_out['prob_pred'] = cls  # batch x n_mod

        return post_out

class Conv1d(nn.Module):
    def __init__(self, n_in, n_out, kernel_size=3, stride=1, norm='GN', ng=32, act=True):
        super(Conv1d, self).__init__()
        assert(norm in ['GN', 'BN', 'SyncBN'])

        self.conv = nn.Conv1d(n_in, n_out, kernel_size=kernel_size, padding=(
            int(kernel_size) - 1) // 2, stride=stride, bias=False)

        if norm == 'GN':
            self.norm = nn.GroupNorm(gcd(ng, n_out), n_out)
        elif norm == 'BN':
            self.norm = nn.BatchNorm1d(n_out)
        else:
            exit('SyncBN has not been added!')

        self.relu = nn.ReLU(inplace=True)
        self.act = act

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        if self.act:
            out = self.relu(out)
        return out

class Res1d(nn.Module):
    def __init__(self, n_in, n_out, kernel_size=3, stride=1, norm='GN', ng=32, act=True):
        super(Res1d, self).__init__()
        assert(norm in ['GN', 'BN', 'SyncBN'])
        padding = (int(kernel_size) - 1) // 2
        self.conv1 = nn.Conv1d(n_in, n_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.conv2 = nn.Conv1d(n_out, n_out, kernel_size=kernel_size, padding=padding, bias=False)
        self.relu = nn.ReLU(inplace=True)

        # All use name bn1 and bn2 to load imagenet pretrained weights
        if norm == 'GN':
            self.bn1 = nn.GroupNorm(gcd(ng, n_out), n_out)
            self.bn2 = nn.GroupNorm(gcd(ng, n_out), n_out)
        elif norm == 'BN':
            self.bn1 = nn.BatchNorm1d(n_out)
            self.bn2 = nn.BatchNorm1d(n_out)
        else:
            exit('SyncBN has not been added!')

        if stride != 1 or n_out != n_in:
            if norm == 'GN':
                self.downsample = nn.Sequential(
                    nn.Conv1d(n_in, n_out, kernel_size=1, stride=stride, bias=False),
                    nn.GroupNorm(gcd(ng, n_out), n_out))
            elif norm == 'BN':
                self.downsample = nn.Sequential(
                    nn.Conv1d(n_in, n_out, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm1d(n_out))
            else:
                exit('SyncBN has not been added!')
        else:
            self.downsample = None

        self.act = act

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            x = self.downsample(x)

        out += x
        if self.act:
            out = self.relu(out)
        return out



class Actor_Encoder_LSTM(nn.Module):
    """
    Actor feature extractor with Conv1D, LSTM, and attention for local coordinate trajectory encoding.
    """

    def __init__(self, device, n_in=3, hidden_size=128, n_fpn_scale=3, lstm_hidden_size=128, lstm_num_layers=1):
        super(Actor_Encoder_LSTM, self).__init__()
        self.device = device
        norm = "GN"
        ng = 1

        # 卷积和残差网络部分（多尺度特征提取）
        n_out = [2**(5 + s) for s in range(n_fpn_scale)]  # [32, 64, 128]
        blocks = [Res1d] * n_fpn_scale
        num_blocks = [2] * n_fpn_scale

        groups = []
        for i in range(len(num_blocks)):
            group = []
            if i == 0:
                group.append(blocks[i](n_in, n_out[i], norm=norm, ng=ng))
            else:
                group.append(blocks[i](n_in, n_out[i], stride=2, norm=norm, ng=ng))
            for j in range(1, num_blocks[i]):
                group.append(blocks[i](n_out[i], n_out[i], norm=norm, ng=ng))
            groups.append(nn.Sequential(*group))
            n_in = n_out[i]
        self.groups = nn.ModuleList(groups)

        lateral = []
        for i in range(len(n_out)):
            lateral.append(Conv1d(n_out[i], hidden_size, norm=norm, ng=ng, act=False))
        self.lateral = nn.ModuleList(lateral)
        self.output = Res1d(hidden_size, hidden_size, norm=norm, ng=ng)

        # LSTM 部分
        self.lstm = nn.LSTM(
            input_size=hidden_size,  # 输入特征维度
            hidden_size=lstm_hidden_size,  # LSTM 隐藏层维度
            num_layers=lstm_num_layers,  # LSTM 层数
            batch_first=True,  # 输入形状为 (batch, seq_len, feature)
            bidirectional=False  # 单向 LSTM
        )
        self.lstm_hidden_size = lstm_hidden_size

        # 注意力机制
        self.attention = nn.Sequential(
            nn.Linear(lstm_hidden_size, lstm_hidden_size),
            nn.Tanh(),
            nn.Linear(lstm_hidden_size, 1)
        )

        # 全连接层，用于将 LSTM 输出映射到目标维度
        self.fc = nn.Linear(lstm_hidden_size, hidden_size)

    def forward(self, actors: Tensor) -> Tensor:
        # 卷积网络部分（多尺度特征提取）
        out = actors
        outputs = []
        for i in range(len(self.groups)):
            out = self.groups[i](out)
            outputs.append(out)

        out = self.lateral[-1](outputs[-1])
        for i in range(len(outputs) - 2, -1, -1):
            out = F.interpolate(out, scale_factor=2, mode="linear", align_corners=False)
            out += self.lateral[i](outputs[i])

        out = self.output(out)  # 输出形状: (N, hidden_size, 20)

        # 将卷积网络输出转换为LSTM输入形状
        out = out.permute(0, 2, 1)  # 形状: (N, 20, hidden_size)

        # LSTM 部分
        lstm_out, _ = self.lstm(out)  # 输出形状: (N, 20, lstm_hidden_size)
        lstm_out = lstm_out.permute(0, 2, 1)
        # print(lstm_out.shape)
        out = self.output(lstm_out)[:, :, -1]

        return out



class Actor_Encoder(nn.Module):
    """
    Actor feature extractor with Conv1D
    """

    def __init__(self, device, n_in=3, hidden_size=128, n_fpn_scale=3):
        # n_fpn_scale变为3，参数在cfg.py中定义net_cfg["n_fpn_scale"] = 3
        super(Actor_Encoder, self).__init__()
        self.device = device
        norm = "GN"
        ng = 1

        n_out = [2**(5 + s) for s in range(n_fpn_scale)]  # [32, 64, 128]
        # 3层残差块: [re, re, re]
        blocks = [Res1d] * n_fpn_scale
        # [2, 2, 2]
        num_blocks = [2] * n_fpn_scale

        groups = []
        # num_blocks长度为3
        for i in range(len(num_blocks)):
            group = []
            # 若为第0层，Res（3, 32）
            if i == 0:
                group.append(blocks[i](n_in, n_out[i], norm=norm, ng=ng))  # i=0, (3,32)->Res1D
            # 若为第其它层，Res（3, 64/128/256）
            else:
                group.append(blocks[i](n_in, n_out[i], stride=2, norm=norm, ng=ng))  # i=1, (32,64)->Res1D; i=2, (64,128)->Res1D

            # num_blocks为[2, 2, 2]
            # j始终为1
            for j in range(1, num_blocks[i]):
                # i=0,（32，32）-> Res1D   若i=1, (64, 64) -> Res1D   若i=2,（128，128）-> Res1D
                group.append(blocks[i](n_out[i], n_out[i], norm=norm, ng=ng))
            # 叠加网络块
            groups.append(nn.Sequential(*group))
            # n_out=32,64,128
            n_in = n_out[i]
        self.groups = nn.ModuleList(groups)

        lateral = []
        # n_out长度为3
        for i in range(len(n_out)):
            # i=0, (32, 128)->Conv1d  i=1, (64, 128)->Conv1d  i=2, (128, 128)->Conv1d
            lateral.append(Conv1d(n_out[i], hidden_size, norm=norm, ng=ng, act=False))
        # 3层卷积网络堆叠
        self.lateral = nn.ModuleList(lateral)
        # 残差网络（128，128）
        self.output = Res1d(hidden_size, hidden_size, norm=norm, ng=ng)

    def forward(self, actors: Tensor) -> Tensor:
        out = actors

        outputs = []
        for i in range(len(self.groups)):
            out = self.groups[i](out)
            outputs.append(out)

        out = self.lateral[-1](outputs[-1])
        for i in range(len(outputs) - 2, -1, -1):
            out = F.interpolate(out, scale_factor=2, mode="linear", align_corners=False)
            out += self.lateral[i](outputs[i])

        out = self.output(out)[:, :, -1]
        # print(out.shape)

        return out

class PointFeatureAggregator(nn.Module):
    def __init__(self, hidden_size: int, aggre_out: bool, dropout: float = 0.1) -> None:
        super(PointFeatureAggregator, self).__init__()
        self.aggre_out = aggre_out

        self.fc1 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(inplace=True)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(inplace=True)
        )
        self.norm = nn.LayerNorm(hidden_size)

    def _global_maxpool_aggre(self, feat):
        # .permute维度交换操作, adaptive_max_pool1d（自适应一维最大池化）沿着感兴趣的维度dim=1, 将输入池化为1
        # permute后沿某维度进行处理时，依旧按照原来索引进行
        return F.adaptive_max_pool1d(feat.permute(0, 2, 1), 1).permute(0, 2, 1)

    def forward(self, x_inp):
        x = self.fc1(x_inp)  # [N_{lane}, 10, hidden_size]
        x_aggre = self._global_maxpool_aggre(x)  # [N, 1, 128]
        # repeat沿着第二个维度复制 x_aggre, x_aggre.repeat([1, x.shape[1], 1])---（N，10，128）

        # cat沿最后一维拼接：（N，10，256）
        x_aggre = torch.cat([x, x_aggre.repeat([1, x.shape[1], 1])], dim=-1)

        # [N_{lane}, 10, hidden_size]
        # （N，10，128）
        out = self.norm(x_inp + self.fc2(x_aggre))
        if self.aggre_out:
            # .squeeze默认沿着维度为1的轴进行压缩，将（N，1，128）--> （N，128）
            return self._global_maxpool_aggre(out).squeeze()
        else:
            return out


class Map_Encoder(nn.Module):
    def __init__(self, device, in_size=10, hidden_size=128, dropout=0.1):
        super(Map_Encoder, self).__init__()
        self.device = device

        self.proj = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(inplace=True)
        )
        self.aggre1 = PointFeatureAggregator(hidden_size=hidden_size, aggre_out=False, dropout=dropout)
        self.aggre2 = PointFeatureAggregator(hidden_size=hidden_size, aggre_out=True, dropout=dropout)
    '''
    def forward(self, feats):
        outs = []
        for feat in feats:
            x = self.proj(feat)  # [N_{lane}, 10, hidden_size]
            x = self.aggre1(x)
            x = self.aggre2(x)  # [N_{lane}, hidden_size]
            outs.extend(x)
        return outs
    '''
    def forward(self, feats):
        x = self.proj(feats)  # [N_{lane}, 10, hidden_size]
        x = self.aggre1(x)  # [N，10，128]
        x = self.aggre2(x)  # [N_{lane}, hidden_size]
        return x

class Spatial_Feature_Layer(nn.Module):
    def __init__(self,
                 device,
                 d_edge: int = 128,
                 d_model: int = 128,
                 d_ffn: int = 2048,
                 n_head: int = 8,
                 dropout: float = 0.1,
                 update_edge: bool = True) -> None:
        super(Spatial_Feature_Layer, self).__init__()
        self.device = device
        self.update_edge = update_edge

        self.proj_memory = nn.Sequential(
            nn.Linear(d_model + d_model + d_edge, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(inplace=True)
        )

        if self.update_edge:
            self.proj_edge = nn.Sequential(
                nn.Linear(d_model, d_edge),
                nn.LayerNorm(d_edge),
                nn.ReLU(inplace=True)
            )
            self.norm_edge = nn.LayerNorm(d_edge)

        self.multihead_attn = MultiheadAttention(
            embed_dim=d_model, num_heads=n_head, dropout=dropout, batch_first=False)

        # Feedforward model
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)

        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = nn.ReLU(inplace=True)

    def forward(self,
                node: Tensor,
                edge: Tensor,
                edge_mask: Optional[Tensor]) -> Tensor:
        '''
            input:
                node:       (N, d_model)
                edge:       (N, N, d_model)
                edge_mask:  (N, N)
        '''
        # update node
        # memory为论文中的C（context），需要拼接源，目标节点，边组成全局上下文，并简单编码---返回（1，N，128），（N，N，128），（N，N，128）
        x, edge, memory = self._build_memory(node, edge)
        # 用节点特征作为q，全局上下文信息作为k，v，做多头注意力（掩码矩阵edge_mask未传入，不对边进行掩盖操作）---返回注意力计算结果（1，N，128）
        x_prime, _ = self._mha_block(x, memory, attn_mask=None, key_padding_mask=edge_mask)
        # 节点特征与注意力结果做残差拼接，并去掉维度为1的轴---（N，128）
        x = self.norm2(x + x_prime).squeeze()
        # 残差连接---（N，128）
        x = self.norm3(x + self._ff_block(x))
        # 返回节点特征矩阵，边矩阵---（N，128），（N，N，128），未对边做更新操作
        return x, edge, None

    def _build_memory(self,
                      node: Tensor,
                      edge: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        '''
            input:
                node:   (N, d_model)
                edge:   (N, N, d_edge)
            output:
                :param  (1, N, d_model)
                :param  (N, N, d_edge)
                :param  (N, N, d_model)
        '''
        n_token = node.shape[0]

        # 1. build memory
        # 将节点矩阵进行扩展，一个在第0维扩展一维，并在其维度复制N个---构建出源节点与目标节点的特征编码矩阵
        src_x = node.unsqueeze(dim=0).repeat([n_token, 1, 1])  # (N, N, d_model)
        tar_x = node.unsqueeze(dim=1).repeat([1, n_token, 1])  # (N, N, d_model)

        # 将源节点，目标节点，相对关系RPE沿着最后一维拼接---（N，N，128+128+128）-> (N, N, 128)，memory为论文中的C（context）
        memory = self.proj_memory(torch.cat([edge, src_x, tar_x], dim=-1))  # (N, N, d_model)

        # 2. (optional) update edge (with residual)
        if self.update_edge:
            edge = self.norm_edge(edge + self.proj_edge(memory))  # (N, N, d_edge)

        return node.unsqueeze(dim=0), edge, memory

    # multihead attention block
    def _mha_block(self,
                   # 多头注意力操作
                   x: Tensor,
                   mem: Tensor,
                   # 后两个参数未传入
                   attn_mask: Optional[Tensor],
                   key_padding_mask: Optional[Tensor]) -> Tensor:
        '''
            input:
                x:                  [1, N, d_model]
                mem:                [N, N, d_model]
                attn_mask:          [N, N]
                key_padding_mask:   [N, N]
            output:
                :param      [1, N, d_model]
                :param      [N, N]
        '''
        # 使用多头注意力，传入（q，k，v），得到注意力计算后的节点信息---[1, N, d_model]
        x, _ = self.multihead_attn(x, mem, mem,
                                   # 后两个参数未传入
                                   attn_mask=attn_mask,
                                   key_padding_mask=key_padding_mask,
                                   need_weights=False)  # return average attention weights
        return self.dropout2(x), None

    # feed forward block
    def _ff_block(self,
                  x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)


class SymmetricFusionTransformer(nn.Module):
    def __init__(self,
                 device,
                 d_model: int = 128,
                 d_edge: int = 128,
                 n_head: int = 8,
                 n_layer: int = 6,
                 dropout: float = 0.1,
                 update_edge: bool = True):
        super(SymmetricFusionTransformer, self).__init__()
        self.device = device

        fusion = []
        for i in range(n_layer):
            # 循环6层融合网络，前5层都需要更新节点，最后一层不更新节点信息
            need_update_edge = False if i == n_layer - 1 else update_edge
            fusion.append(Spatial_Feature_Layer(device=device,
                                   d_edge=d_edge,
                                   d_model=d_model,
                                   d_ffn=d_model*2,
                                   n_head=n_head,
                                   dropout=dropout,
                                   update_edge=need_update_edge))
        self.fusion = nn.ModuleList(fusion)

    def forward(self, x: Tensor, edge: Tensor, edge_mask: Tensor) -> Tensor:
        '''
            x: (N, d_model)
            edge: (d_model, N, N)
            edge_mask: (N, N)
        '''
        # attn_multilayer = []
        for mod in self.fusion:
            x, edge, _ = mod(x, edge, edge_mask)
            # attn_multilayer.append(attn)
        return x, None


class FusionNet(nn.Module):
    def __init__(self, device, config):
        super(FusionNet, self).__init__()
        self.device = device
        d_embed = config['d_embed']  # 128
        dropout = config['dropout']
        # True
        update_edge = config['update_edge']  # True

        self.proj_actor = nn.Sequential(
            # 128，128
            nn.Linear(config['d_actor'], d_embed),
            nn.LayerNorm(d_embed),
            nn.ReLU(inplace=True)
        )
        self.proj_lane = nn.Sequential(
            # 128，128
            nn.Linear(config['d_lane'], d_embed),
            nn.LayerNorm(d_embed),
            nn.ReLU(inplace=True)
        )
        self.proj_rpe_scene = nn.Sequential(
            # 5，128
            nn.Linear(config['d_rpe_in'], config['d_rpe']),
            nn.LayerNorm(config['d_rpe']),
            nn.ReLU(inplace=True)
        )

        self.fuse_scene = SymmetricFusionTransformer(self.device,
                                                     # 128，128，8，4
                                                     d_model=d_embed,
                                                     d_edge=config['d_rpe'],
                                                     n_head=config['n_scene_head'],
                                                     n_layer=config['n_scene_layer'],
                                                     dropout=dropout,
                                                     update_edge=update_edge)

    def forward(self,
                actors: Tensor,
                actor_idcs: List[Tensor],
                lanes: Tensor,
                lane_idcs: List[Tensor],
                rpe_prep: Dict[str, Tensor]):
        # 传参：车辆编码，对应车辆id，车道编码，对应车道id，相对位置RPE
        # print('actors: ', actors.shape)
        # print('actor_idcs: ', [x.shape for x in actor_idcs])
        # print('lanes: ', lanes.shape)
        # print('lane_idcs: ', [x.shape for x in lane_idcs])

        # projection
        # 车辆与车道编码进行维度对齐
        actors = self.proj_actor(actors)
        lanes = self.proj_lane(lanes)

        # 初始化更新列表
        actors_new, lanes_new = list(), list()
        # 对每个批次中的信息进行处理
        for a_idcs, l_idcs, rpes in zip(actor_idcs, lane_idcs, rpe_prep):
            # 在一个batch内，各场景文件中的车辆id，车道id，相对边关系
            # * fusion - scene
            # 根据对应id找到该车辆/车道信息的编码---（N，128），(M, 128)
            _actors = actors[a_idcs]
            _lanes = lanes[l_idcs]

            # 根据索引取出车辆和车道特征，以0维纵向拼接（N+M, 128）
            tokens = torch.cat([_actors, _lanes], dim=0)  # (N+M, d_model)
            # 第一次转系坐标系下，元素间相对关系RPE（5，N + M，N + M）--->（N+M，N+M，5）

            # 相对关系编码，rep---(N+M, N+M, 128)，N为车辆数和车道数的总数
            rpe = self.proj_rpe_scene(rpes['scene'].permute(1, 2, 0))  # (N+M, N+M, d_rpe)

            # 返回锚姿态更新后的节点特征
            # token中的一对节点，通过rpe更新其节点特征, mask在预处理时候没有传入值---返回更新后的节点特征编码（N，128）
            out, _ = self.fuse_scene(tokens, rpe, rpes['scene_mask'])

            # 前一部分是车辆特征，后一部分是车道特征，分别赋值给车辆与车道，作为节点更新操作
            actors_new.append(out[:len(a_idcs)])
            lanes_new.append(out[len(a_idcs):])
        # print('actors: ', [x.shape for x in actors_new])
        # print('lanes: ', [x.shape for x in lanes_new])

        # 分别拼接每个批次中的各场景更新后的车辆节点和车道节点特征
        actors = torch.cat(actors_new, dim=0)
        lanes = torch.cat(lanes_new, dim=0)
        # print('actors: ', actors.shape)
        # print('lanes: ', lanes.shape)

        # 返回SIMPL通过SFT更新后的车辆与车道编码，simpl直接在本部分完成后进行预测，GUP会有后续处理
        return actors, lanes, None

class Interaction_Module_SC_AM(nn.Module):
    # 输入（batch, max_agent_num, d128），（batch, max_lane_num，d128）,mask
    def __init__(self, device, hidden_size, depth=3):
        super(Interaction_Module_SC_AM, self).__init__()

        self.depth = depth

        # 每部分作3层注意力
        self.AA = nn.ModuleList([Attention_Block(hidden_size) for _ in range(depth)])
        self.AL = nn.ModuleList([Attention_Block(hidden_size) for _ in range(depth)])
        self.LL = nn.ModuleList([Attention_Block(hidden_size) for _ in range(depth)])
        self.LA = nn.ModuleList([Attention_Block(hidden_size) for _ in range(depth)])

    def forward(self, agent_features, lane_features, masks):
        # 输入（batch, max_agent_num, d128），（batch, max_lane_num，d128）,mask
        # 按照mask倒序进行l2a, l2l, a2l, a2a
        for layer_index in range(self.depth):
            # === Lane to Agent ===
            lane_features = self.LA[layer_index](lane_features, agent_features, attn_mask=masks[-1])
            # === === ===

            # === Lane to Lane ===
            lane_features = self.LL[layer_index](lane_features, attn_mask=masks[-2])
            # === ==== ===

            # === Agent to Lane ===
            agent_features = self.AL[layer_index](agent_features, lane_features, attn_mask=masks[-3])
            # === ==== ===

            # === Agent to Agent ===
            agent_features = self.AA[layer_index](agent_features, attn_mask=masks[-4])
            # === ==== ===

        # 返回更新后的车辆，车道特征
        return agent_features, lane_features


class Interaction_Module_FE(nn.Module):
    def __init__(self, device, hidden_size, depth=3):
        super(Interaction_Module_FE, self).__init__()

        self.depth = depth
        self.AL = nn.ModuleList([Attention_Block(hidden_size) for _ in range(depth)])


    def forward(self, agent_features, lane_features, masks):

        for layer_index in range(self.depth):

            # === Agent to Lane ===
            agent_features = self.AL[layer_index](agent_features, lane_features, attn_mask=masks[0])
            # === ==== ===

        return agent_features, lane_features

class Interaction_Module_M2M(nn.Module):
    # 256
    def __init__(self, hidden_size, depth=3):
        super(Interaction_Module_M2M, self).__init__()

        # 3层注意力机制
        self.depth = depth
        self.MM = nn.ModuleList([M2M_Attention_Block(hidden_size) for _ in range(depth)])


    def forward(self, mode1_features, mode2_features):

        for layer_index in range(self.depth):

            # === Agent to Lane ===
            mode1_features = self.MM[layer_index](mode1_features, mode2_features)
            # === ==== ===

        return mode1_features


class Attention_Block(nn.Module):
    # 8头注意力机制
    def __init__(self, hidden_size, num_heads=8, p_drop=0.1):

        super(Attention_Block, self).__init__()
        self.multiheadattention = Attention(hidden_size, num_heads, p_drop)

        self.ffn_layer = MLP(hidden_size, hidden_size)

        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)

    def forward(self, query, key_value=None, attn_mask=None):
        # 交叉注意力与自注意力
        if key_value is None:
            key_value = query

        attn_output = self.multiheadattention(
            query, key_value, attention_mask=attn_mask)

        query = self.norm1(attn_output + query)
        query_temp = self.ffn_layer(query)
        query = self.norm2(query_temp + query)

        return query

class M2M_Attention_Block(nn.Module):
    # 8头注意力机制
    def __init__(self, hidden_size, num_heads=8, p_drop=0.1):

        super(M2M_Attention_Block, self).__init__()
        self.multiheadattention = M2M_Attention(hidden_size, num_heads, p_drop)

        self.ffn_layer = MLP(hidden_size, hidden_size)

        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)

    def forward(self, query, key_value=None, attn_mask=None):
        # 交叉注意力与自注意力
        if key_value is None:
            key_value = query

        attn_output = self.multiheadattention(
            query, key_value, attention_mask=attn_mask)

        query = self.norm1(attn_output + query)
        query_temp = self.ffn_layer(query)
        query = self.norm2(query_temp + query)
        return query

class Attention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, p_drop):
        super(Attention, self).__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)
        self.last_projection = nn.Linear(self.all_head_size, hidden_size)
        self.attention_drop = nn.Dropout(p_drop)

    def get_extended_attention_mask(self, attention_mask):
        extended_attention_mask = attention_mask.unsqueeze(1)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def transpose_for_scores(self, x):
        sz = x.size()[:-1] + (self.num_attention_heads,
                              self.attention_head_size)
        # (batch, max_vector_num, head, head_size)
        x = x.view(*sz)
        # (batch, head, max_vector_num, head_size)
        return x.permute(0, 2, 1, 3)

    def forward(self, query_states, key_value_states, attention_mask):
        mixed_query_layer = self.query(query_states)
        mixed_key_layer = F.linear(key_value_states, self.key.weight)
        mixed_value_layer = self.value(key_value_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # 注意力计算开始
        # 缩放点积注意力公式体现：(Q*KT)/sqrt(dimk)
        attention_scores = torch.matmul(
            query_layer/math.sqrt(self.attention_head_size), key_layer.transpose(-1, -2))

        if attention_mask is not None:
            attention_scores = attention_scores + \
                self.get_extended_attention_mask(attention_mask)

        # softmax((Q*KT)/sqrt(dimk))
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.attention_drop(attention_probs)

        assert torch.isnan(attention_probs).sum() == 0
        # softmax((Q*KT)/sqrt(dimk))*V
        context_layer = torch.matmul(attention_probs, value_layer)
        # 注意力计算完成

        # .permute维度交换，.contiguous()返回一个在内存中连续的Tensor
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()

        new_context_layer_shape = context_layer.size()[
                                  :-2] + (self.all_head_size,)
        # context_layer.shape = (batch, max_vector_num, all_head_size)
        context_layer = context_layer.view(*new_context_layer_shape)
        context_layer = self.last_projection(context_layer)
        return context_layer

class M2M_Attention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, p_drop):
        super(M2M_Attention, self).__init__()
        # 8，256/8=32，8*32=256
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)
        self.last_projection = nn.Linear(self.all_head_size, hidden_size)
        self.attention_drop = nn.Dropout(p_drop)

    def get_extended_attention_mask(self, attention_mask):
        extended_attention_mask = attention_mask.unsqueeze(1)  # (B, 1, N, 6)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def transpose_for_scores(self, x):
        # Reshape to (B, N, 6, num_attention_heads, attention_head_size)
        # sz = (8,36,6,8,6,32)
        sz = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*sz)
        # Transpose to (B, num_attention_heads, 6, N, attention_head_size)
        return x.permute(0, 3, 1, 2, 4)

    def forward(self, query_states, key_value_states, attention_mask):
        # Mixed query, key, value layers
        # (B, N, 6, 256)
        mixed_query_layer = self.query(query_states)  # (B, N, 6, all_head_size)
        mixed_key_layer = F.linear(key_value_states, self.key.weight)  # (B, N, 6, all_head_size)
        mixed_value_layer = self.value(key_value_states)  # (B, N, 6, all_head_size)

        # Transpose query, key, value layers
        # mixed_query_layer---(B, N, 6, 256)
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Scaled dot-product attention (Q * K^T) / sqrt(d_k)
        attention_scores = torch.matmul(
            query_layer / math.sqrt(self.attention_head_size), key_layer.transpose(-1, -2)
        )

        # If an attention mask is provided, add it to the attention scores
        if attention_mask is not None:
            attention_scores = attention_scores + self.get_extended_attention_mask(attention_mask)

        # Softmax to get attention probabilities
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.attention_drop(attention_probs)

        # Check for NaN in attention_probs
        assert torch.isnan(attention_probs).sum() == 0

        # Attention-weighted sum of values
        context_layer = torch.matmul(attention_probs, value_layer)  # (B, num_attention_heads, N, 6, attention_head_size)

        # Reshape the context_layer to (B, N, 6, all_head_size)
        context_layer = context_layer.permute(0, 2, 3, 1, 4).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        # Final linear projection to original hidden size
        context_layer = self.last_projection(context_layer)
        return context_layer


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, p_drop=0.0, hidden_dim=None, residual=False):
        super(MLP, self).__init__()

        if hidden_dim is None:
            hidden_dim = input_dim

        layer2_dim = hidden_dim
        if residual:
            layer2_dim = hidden_dim + input_dim

        self.residual = residual
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(layer2_dim, output_dim)
        self.dropout1 = nn.Dropout(p=p_drop)
        self.dropout2 = nn.Dropout(p=p_drop)

        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        out = self.layer1(x)
        out = self.norm(out)
        out = F.relu(out)
        out = self.dropout1(out)
        if self.residual:
            out = self.layer2(torch.cat([out, x], dim=-1))
        else:
            out = self.layer2(out)

        out = self.dropout2(out)
        return out

class Attention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, p_drop):
        super(Attention, self).__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)
        self.last_projection = nn.Linear(self.all_head_size, hidden_size)
        self.attention_drop = nn.Dropout(p_drop)

    def get_extended_attention_mask(self, attention_mask):
        extended_attention_mask = attention_mask.unsqueeze(1)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def transpose_for_scores(self, x):
        sz = x.size()[:-1] + (self.num_attention_heads,
                              self.attention_head_size)
        # (batch, max_vector_num, head, head_size)
        x = x.view(*sz)
        # (batch, head, max_vector_num, head_size)
        return x.permute(0, 2, 1, 3)

    def forward(self, query_states, key_value_states, attention_mask):
        mixed_query_layer = self.query(query_states)
        mixed_key_layer = F.linear(key_value_states, self.key.weight)
        mixed_value_layer = self.value(key_value_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # 注意力计算开始
        # 缩放点积注意力公式体现：(Q*KT)/sqrt(dimk)
        attention_scores = torch.matmul(
            query_layer/math.sqrt(self.attention_head_size), key_layer.transpose(-1, -2))

        if attention_mask is not None:
            attention_scores = attention_scores + \
                self.get_extended_attention_mask(attention_mask)

        # softmax((Q*KT)/sqrt(dimk))
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.attention_drop(attention_probs)

        assert torch.isnan(attention_probs).sum() == 0
        # softmax((Q*KT)/sqrt(dimk))*V
        context_layer = torch.matmul(attention_probs, value_layer)
        # 注意力计算完成

        # .permute维度交换，.contiguous()返回一个在内存中连续的Tensor
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()

        new_context_layer_shape = context_layer.size()[
                                  :-2] + (self.all_head_size,)
        # context_layer.shape = (batch, max_vector_num, all_head_size)
        context_layer = context_layer.view(*new_context_layer_shape)
        context_layer = self.last_projection(context_layer)
        return context_layer

class Trajectory_Decoder_Future_Enhanced(nn.Module):
    def __init__(self, device, hidden_size):
        super(Trajectory_Decoder_Future_Enhanced, self).__init__()
        self.endpoint_predictor = MLP(hidden_size, 6*2, residual=True)

        self.get_trajectory = MLP(hidden_size + 2, 29*2, residual=True)
        self.endpoint_refiner = MLP(hidden_size + 2, 2, residual=True)
        #self.get_prob = MLP(hidden_size + 2, 1, residual=True)

    def forward(self, agent_features):
        # 输入维度agent_features.shape = (B, N, 128)
        N = agent_features.shape[0]
        M = agent_features.shape[1]
        D = agent_features.shape[2]

        # 所有车辆末端点解码
        # endpoints.shape = (B, N, 6, 2)
        endpoints = self.endpoint_predictor(agent_features).view(N, M, 6, 2)
        # 在第二维增加一维，重复6次
        # prediction_features.shape = (B, N, 6, 128)
        agent_features_expanded = agent_features.unsqueeze(dim=2).expand(N, M, 6, D)

        # agent_features_expanded 与 endpoints最后一位拼接（B, N, 6, 128+2），经过MLP维度--> (B, N, 6, 2)
        offsets = self.endpoint_refiner(torch.cat([agent_features_expanded, endpoints.detach()], dim=-1))

        # 将两部分末端点相加---（B，N，6，2）
        endpoints += offsets

        # 再次拼接末端点
        # agent_features_expanded.shape = (B, N, 6, 128 + 2)
        agent_features_expanded = torch.cat([agent_features_expanded, endpoints.detach()], dim=-1)

        # 解码出未来轨迹（不包含末端点）---（B，N，6，29，2）
        predictions = self.get_trajectory(agent_features_expanded).view(N, M, 6, 29, 2)
        #logits = self.get_prob(agent_features_expanded).view(N, M, 6)

        # 拼接末端点，cat：（B，N，6，29，2）and（B，N，6，1，2）产生未来多模态轨迹---（B，N，6，30，2）
        predictions = torch.cat([predictions, endpoints.unsqueeze(dim=-2)], dim=-2)

        # 断言函数，若满足不输出，继续执行下一步
        assert predictions.shape == (N, M, 6, 30, 2)

        return predictions



class Trajectory_Decoder_OCC(nn.Module):
    def __init__(self, device, hidden_size):
        super(Trajectory_Decoder_OCC, self).__init__()
        self.endpoint_predictor = MLP(hidden_size, 5*2, residual=True)

        self.get_trajectory = MLP(hidden_size + 2, 29*2, residual=True)
        self.endpoint_refiner = MLP(hidden_size + 2, 5*2, residual=True)
        #self.get_prob = MLP(hidden_size + 2, 1, residual=True)

    def forward(self, agent_features):
        # 输入维度agent_features.shape = (B, N, 128)
        N = agent_features.shape[0]
        M = agent_features.shape[1]
        D = agent_features.shape[2]

        # 所有车辆末端点解码
        # endpoints.shape = (B, N, 6, 2)
        # endpoints = self.endpoint_predictor(agent_features).view(N, M, 6, 2)
        # 在第二维增加一维，重复6次
        # prediction_features.shape = (B, N, 6, 128)
        agent_features_expanded = agent_features.unsqueeze(dim=2).expand(N, M, 6, D)

        # 解码关键帧5步的车道占用情况---(B, N, 6, 5, 2)
        agent_5 = self.endpoint_predictor(agent_features_expanded).view(N, M, 6, 5, 2)
        # 拿出末端点---(B, N, 6, 2)
        endpoints = agent_5[:, :, :, -1, :]

        # agent_features_expanded 与 endpoints最后一位拼接（B, N, 6, 128+2），经过MLP维度--> (B, N, 6, 10)->(B, N, 6, 5, 2)
        offsets = self.endpoint_refiner(torch.cat([agent_features_expanded, endpoints.detach()], dim=-1)).view(N, M, 6, 5, 2)
        offsets = offsets[:, :, :, -1, :]
        #print(endpoints.shape, offsets.shape)
        # 将两部分第5步末端点相加---（B，N，6, 2）
        endpoints += offsets
        # 返回末端点

        # 再次拼接末端点
        # agent_features_expanded.shape = (B, N, 6, 128 + 2)
        #agent_features_expanded = torch.cat([agent_features_expanded, endpoints.detach()], dim=-1)

        # 解码出未来轨迹（不包含末端点）---（B，N，6，29，2）
        #predictions = self.get_trajectory(agent_features_expanded).view(N, M, 6, 29, 2)
        #logits = self.get_prob(agent_features_expanded).view(N, M, 6)

        # 拼接末端点，cat：（B，N，6，29，2）and（B，N，6，1，2）产生未来多模态轨迹---（B，N，6，30，2）
        #predictions = torch.cat([predictions, endpoints.unsqueeze(dim=-2)], dim=-2)

        # 断言函数，若满足不输出，继续执行下一步
        assert endpoints.shape == (N, M, 6, 2)

        return endpoints










class Trajectory_Decoder_Final(nn.Module):
    def __init__(self, device, hidden_size):
        super(Trajectory_Decoder_Final, self).__init__()
        self.endpoint_predictor = MLP(hidden_size, 6*2, residual=True)

        self.get_trajectory = MLP(hidden_size + 2, 29*2, residual=True)
        self.endpoint_refiner = MLP(hidden_size + 2, 2, residual=True)

        self.cls = nn.Sequential(
            nn.Linear(hidden_size + 2, hidden_size + 2),
            nn.LayerNorm(hidden_size+ 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size + 2, hidden_size + 2),
            nn.LayerNorm(hidden_size + 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size + 2, 1)
        )

    def forward(self, agent_features):
        # agent_features.shape = (B, N, 128)
        N = agent_features.shape[0]
        M = agent_features.shape[1]
        D = agent_features.shape[2]

        # 解码多模态轨迹末端点-endpoints.shape = (B, N, 6, 2)
        endpoints = self.endpoint_predictor(agent_features).view(N, M, 6, 2)

        # 维度增加，并沿指定维度重复6次（B，N，128） => (B, N, 6, 128)
        agent_features_expanded = agent_features.unsqueeze(dim=2).expand(N, M, 6, D)

        # 拼接末端点，（B，N，6，128+2）-> offsets.shape = (B, N, 6, 2)
        offsets = self.endpoint_refiner(torch.cat([agent_features_expanded, endpoints.detach()], dim=-1))
        # 拼接末端偏移量
        endpoints += offsets

        # 轨迹特征拼接末端点，agent_features_expanded.shape = (B, N, 6, 128 + 2)
        agent_features_expanded = torch.cat([agent_features_expanded, endpoints.detach()], dim=-1)

        # 解码除末端点以外的轨迹---（B，N，6，29，2）
        predictions = self.get_trajectory(agent_features_expanded).view(N, M, 6, 29, 2)

        #logits = self.get_prob(agent_features_expanded).view(N, M, 6)

        # 概率分数
        # (B, N, 6, 128 + 2) -> (B, N, 6, 1) -> (B, N, 6)
        logits = self.cls(agent_features_expanded).view(N, M, 6)
        # 计算归一化分数-softmax---（B，N，6）
        logits = F.softmax(logits * 1.0, dim=2)  # e.g., [159, 6]

        # 将轨迹前29步与末端点进行拼接（B，N，6，29，2）+（B，N，6，1，2）--> 得完整轨迹（B，N，6，30，2）
        predictions = torch.cat([predictions, endpoints.unsqueeze(dim=-2)], dim=-2)

        # 断言函数，确定形状是否相同
        assert predictions.shape == (N, M, 6, 30, 2)

        # 返回预测轨迹和分数---(B，N，6，30，2)，（B，N，6）
        return predictions, logits


# K个带残差的独立MLP
class Differentiation_Decoder(nn.Module):
    def __init__(self, device, hidden_size, num_modes):
        super(Differentiation_Decoder, self).__init__()
        self.num_modes = num_modes
        self.endpoint_predictor = MLP(hidden_size, 6*2, residual=True)


        # self.get_trajectory = MLP(hidden_size + 2, 29*2, residual=True)
        # self.get_trajectory = nn.ModuleList([MLP(hidden_size, 29*2, residual=True) for _ in range(num_modes)])
        self.get_trajectory = MLP(hidden_size, 29 * 2, residual=True)
        self.endpoint_refiner = MLP(hidden_size + 2, 2, residual=True)
        self.mlp = MLP(hidden_size + 2, hidden_size, residual=True)
        # hiddze = 256
        self.interaction_module_m2m = Interaction_Module_M2M(hidden_size=hidden_size,
                                                             depth=2)

        self.cls = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.LayerNorm(hidden_size * 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size * 2, hidden_size * 2),
            nn.LayerNorm(hidden_size * 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size * 2, 1)
        )

    def forward(self, agent_features, global_map_embed):
        # agent_features.shape = (B, N, 256)
        N = agent_features.shape[0]
        M = agent_features.shape[1]
        D = agent_features.shape[2]

        # 解码多模态轨迹末端点-endpoints.shape = (B, N, 6, 2)
        endpoints = self.endpoint_predictor(agent_features).view(N, M, 6, 2)
        # 维度增加，并沿指定维度重复6次（B，N，256） => (B, N, 6, 256)
        agent_features_expanded = agent_features.unsqueeze(dim=2).expand(N, M, 6, D)
        # 拼接末端点，（B，N，6，128+2）-> offsets.shape = (B, N, 6, 2)
        offsets = self.endpoint_refiner(torch.cat([agent_features_expanded, endpoints.detach()], dim=-1))
        # 拼接末端偏移量
        endpoints += offsets


        # 轨迹特征拼接末端点，agent_features_expanded.shape = (B, N, 6, 128 + 2)
        agent_features_expanded = torch.cat([agent_features_expanded, endpoints.detach()], dim=-1)
        # 变为(B, N, 6, 256)
        agent_features_expanded = self.mlp(agent_features_expanded)

        # 执行模式查询M2M注意力机制---(B, N, 6, 128)
        agent_mode_updated = self.interaction_module_m2m(agent_features_expanded, global_map_embed)

        # 多模态列表初始化
        agent_mode = []
        # layer_index = 6
        for layer_index in range(self.num_modes):
            # 解码单模态下的预测结果---(B, N, 256)-->(B, N, 29*2)-->(B, N, 29, 2)
            agent_features = self.get_trajectory(agent_mode_updated[:, :, layer_index, :])
            # 形状对齐
            agent_features = agent_features.view(N, M, 29, 2)
            # 追加多模态列表
            agent_mode.append(agent_features)
        # 按照0维拼接---(6, B, N, 29, 2)
        agent_mode = torch.stack(agent_mode)

        # 维度交换---（B，N，6，29，2）
        agent_mode = agent_mode.permute(1, 2, 0, 3, 4)

        # 拼接其末端点---（B，N，6，30，2）
        predictions = torch.cat([agent_mode, endpoints.unsqueeze(dim=-2)], dim=-2)
        #predictions = torch.cat([agent_mode, agent_mode[:, :, :, -1, :].unsqueeze(dim=-2)], dim=-2)

        # 概率分数
        # (B, N, 6, 128 + 2) -> (B, N, 6, 1) -> (B, N, 6)
        logits = self.cls(agent_mode_updated).view(N, M, 6)
        # 计算归一化分数-softmax---（B，N，6）
        logits = F.softmax(logits * 1.0, dim=2)  # e.g., [159, 6]

        # 断言函数，确定形状是否相同
        assert predictions.shape == (N, M, 6, 30, 2)

        # 返回预测轨迹和分数---(B，N，6，30，2)，（B，N，6）
        return predictions, logits


class Trajectory_Decoder_All(nn.Module):
    def __init__(self, device, hidden_size):
        super(Trajectory_Decoder_All, self).__init__()
        self.endpoint_predictor = MLP(hidden_size, 6*2, residual=True)

        self.get_trajectory = MLP(hidden_size + 2, 29*2, residual=True)
        self.endpoint_refiner = MLP(hidden_size + 2, 2, residual=True)

        self.cls = nn.Sequential(
            nn.Linear(hidden_size + 2, hidden_size + 2),
            nn.LayerNorm(hidden_size+ 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size + 2, hidden_size + 2),
            nn.LayerNorm(hidden_size + 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size + 2, 1)
        )

    def forward(self, agent_features):
        # agent_features.shape = (B, N, 128*4)
        N = agent_features.shape[0]
        M = agent_features.shape[1]
        D = agent_features.shape[2]

        # 解码多模态轨迹末端点-endpoints.shape = (B, N, 6, 2)
        endpoints = self.endpoint_predictor(agent_features).view(N, M, 6, 2)

        # 维度增加，并沿指定维度重复6次（B，N，128） => (B, N, 6, 128)
        agent_features_expanded = agent_features.unsqueeze(dim=2).expand(N, M, 6, D)

        # 拼接末端点，（B，N，6，128+2）-> offsets.shape = (B, N, 6, 2)
        offsets = self.endpoint_refiner(torch.cat([agent_features_expanded, endpoints.detach()], dim=-1))
        # 拼接末端偏移量
        endpoints += offsets

        # 轨迹特征拼接末端点，agent_features_expanded.shape = (B, N, 6, 128 + 2)
        agent_features_expanded = torch.cat([agent_features_expanded, endpoints.detach()], dim=-1)

        # 解码除末端点以外的轨迹---（B，N，6，29，2）
        predictions = self.get_trajectory(agent_features_expanded).view(N, M, 6, 29, 2)

        #logits = self.get_prob(agent_features_expanded).view(N, M, 6)

        # 概率分数
        # (B, N, 6, 128 + 2) -> (B, N, 6, 1) -> (B, N, 6)
        logits = self.cls(agent_features_expanded).view(N, M, 6)
        # 计算归一化分数-softmax---（B，N，6）
        logits = F.softmax(logits * 1.0, dim=2)  # e.g., [159, 6]

        # 将轨迹前29步与末端点进行拼接（B，N，6，29，2）+（B，N，6，1，2）--> 得完整轨迹（B，N，6，30，2）
        predictions = torch.cat([predictions, endpoints.unsqueeze(dim=-2)], dim=-2)

        # 断言函数，确定形状是否相同
        assert predictions.shape == (N, M, 6, 30, 2)

        # 返回预测轨迹和分数---(B，N，6，30，2)，（B，N，6）
        return predictions, logits




class Reliable_Future_Trajectory_Encoder(nn.Module):
    def __init__(self):
        super(Reliable_Future_Trajectory_Encoder, self).__init__()
        self.get_encoder = MLP(360, 128, residual=True)

    def forward(self, agent_features):
        # 输入---（B，N，6，5，2）
        N, M, _, _, _ = agent_features.shape

        # 调整未来轨迹的特征维度（B， N， 6， 30， 2）-->（B， N， 360）
        flattened_input = agent_features.view(N, M, -1)

        # 送入MLP中进行维度编码---（B，N，128）
        future_features = self.get_encoder(flattened_input).view(N, M, 128)

        return future_features


class Reliable_Future_Trajectory_Encoder1(nn.Module):
    def __init__(self):
        super(Reliable_Future_Trajectory_Encoder1, self).__init__()
        self.get_encoder = MLP(60, 128, residual=True)

    def forward(self, agent_features):
        # 输入---（B，N，6，5，2）
        N, M, _, _, _ = agent_features.shape

        # 调整未来轨迹的特征维度（B， N， 6， 30， 2）-->（B， N， 360）
        flattened_input = agent_features.reshape(N, M, -1)

        # 送入MLP中进行维度编码---（B，N，128）
        future_features = self.get_encoder(flattened_input).view(N, M, 128)

        return future_features


class Reliable_Future_Trajectory_Encoder2(nn.Module):
    def __init__(self):
        super(Reliable_Future_Trajectory_Encoder2, self).__init__()
        self.get_encoder = MLP(300, 128, residual=True)

    def forward(self, agent_features):
        # 输入---（B，N，6，5，2）
        N, M, _, _, _ = agent_features.shape

        # 调整未来轨迹的特征维度（B， N， 6， 30， 2）-->（B， N， 360）
        flattened_input = agent_features.reshape(N, M, -1)

        # 送入MLP中进行维度编码---（B，N，128）
        future_features = self.get_encoder(flattened_input).view(N, M, 128)

        return future_features





class Occupancy_Encoder(nn.Module):
    def __init__(self):
        super(Occupancy_Encoder, self).__init__()
        self.get_encoder = MLP(256, 128, residual=True)

    def forward(self, agent_features):
        # 输入---（B，N，257）
        N, M, _ = agent_features.shape

        # 调整未来轨迹的特征维度（B， N， 257）-->（B， N， 128）
        flattened_input = agent_features
        # 送入MLP中进行维度编码---（B，N，128）
        future_features = self.get_encoder(flattened_input).view(N, M, 128)

        return future_features





def get_mask(agent_lengths, lane_lengths, device):
    # 找出具有最大数量车辆和车道的场景
    max_lane_num = max(lane_lengths)
    max_agent_num = max(agent_lengths)
    batch_size = len(agent_lengths)

    # === Agent - Lane Mask ===
    # query: agent, key-value: lane
    AL_mask = torch.zeros(
        batch_size, max_agent_num, max_lane_num, device=device)

    for i, (agent_length, lane_length) in enumerate(zip(agent_lengths, lane_lengths)):
        AL_mask[i, :agent_length, :lane_length] = 1

    masks = [AL_mask]

    # === === === === ===
    return masks

def get_mask_l2a(lane_lengths, agent_lengths, device):
    max_lane_num = max(lane_lengths)
    max_agent_num = max(agent_lengths)
    # 批次数
    batch_size = len(agent_lengths)

    # === Lane - Agent Mask ===
    # query: lane, key-value: agent
    LA_mask = torch.zeros(
        batch_size, max_lane_num, max_agent_num, device=device)

    for i, (lane_length, agent_length) in enumerate(zip(lane_lengths, agent_lengths)):
        LA_mask[i, :lane_length, :agent_length] = 1

    masks = [LA_mask]
    return masks



def get_masks(agent_lengths, lane_lengths, device):
    # 输入一个批次各场景的车辆个数，车道个数
    # 找出具有最多车辆和车道的场景
    max_lane_num = max(lane_lengths)
    max_agent_num = max(agent_lengths)
    # 批次数
    batch_size = len(agent_lengths)

    # === === Mask Generation Part === ===
    # === Agent - Agent Mask ===
    # query: agent, key-value: agent
    # 初始化车车遮蔽矩阵
    AA_mask = torch.zeros(batch_size, max_agent_num, max_agent_num, device=device)

    for i, agent_length in enumerate(agent_lengths):
        # 单场景下，若车辆编码有效，置1，无效为0
        AA_mask[i, :agent_length, :agent_length] = 1
    # === === ===

    # === Agent - Lane Mask ===
    # query: agent, key-value: lane
    # 初始化车-车道遮蔽矩阵
    AL_mask = torch.zeros(batch_size, max_agent_num, max_lane_num, device=device)

    for i, (agent_length, lane_length) in enumerate(zip(agent_lengths, lane_lengths)):
        # 单场景下，若车，车道编码有效，置1，无效为0
        AL_mask[i, :agent_length, :lane_length] = 1
    # === === ===

    # === Lane - Lane Mask ===
    # query: lane, key-value: lane
    LL_mask = torch.zeros(batch_size, max_lane_num, max_lane_num, device=device)

    QL_mask = torch.zeros(batch_size, 6, max_lane_num, device=device)

    for i, lane_length in enumerate(lane_lengths):
        LL_mask[i, :lane_length, :lane_length] = 1

        QL_mask[i, :, :lane_length] = 1

    # === === ===

    # === Lane - Agent Mask ===
    # query: lane, key-value: agent
    LA_mask = torch.zeros(
        batch_size, max_lane_num, max_agent_num, device=device)

    for i, (lane_length, agent_length) in enumerate(zip(lane_lengths, agent_lengths)):
        LA_mask[i, :lane_length, :agent_length] = 1

    # === === ===

    # 返回所有遮蔽矩阵，后边进行一次转系下的四层注意力机制
    masks = [AA_mask, AL_mask, LL_mask, LA_mask]

    # === === === === ===
    # QL_mask未用到
    return masks, QL_mask


class MultiModalTrajectoryDecoder(nn.Module):
    def __init__(self, feature_dim, hidden_dim, num_heads, num_layers, num_modes, future_steps, output_dim):
        super(MultiModalTrajectoryDecoder, self).__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_modes = num_modes
        self.future_steps = future_steps
        self.output_dim = output_dim

        # 多头注意力层
        self.attention = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=num_heads)

        # 6 个独立的 LSTM 解码器
        self.rnns = nn.ModuleList([
            nn.LSTM(input_size=feature_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
            for _ in range(num_modes)
        ])

        # 6 个独立的输出层
        self.fcs = nn.ModuleList([
            nn.Linear(hidden_dim, output_dim * future_steps)  # 输出 future_steps 步的轨迹
            for _ in range(num_modes)
        ])

        # 轨迹概率值生成层
        self.prob_fc = nn.Linear(hidden_dim * num_modes, num_modes)  # 输入是所有解码器的隐藏状态

    def forward(self, vehicle_features, lane_features):
        """
        vehicle_features: (B, N, D) - 车辆特征
        lane_features: (B, M, D) - 车道特征
        """
        B, N, D = vehicle_features.shape
        M = lane_features.shape[1]

        # 初始化隐藏状态
        h = torch.zeros(self.num_layers, B * N, self.hidden_dim).to(vehicle_features.device)
        c = torch.zeros(self.num_layers, B * N, self.hidden_dim).to(vehicle_features.device)

        # 存储预测结果和隐藏状态
        predictions = []
        hidden_states = []

        # 对每条轨迹进行解码
        for mode in range(self.num_modes):
            # 计算注意力
            query = vehicle_features.transpose(0, 1)  # (N, B, D)
            key = lane_features.transpose(0, 1)       # (M, B, D)
            value = lane_features.transpose(0, 1)     # (M, B, D)

            attn_output, _ = self.attention(query, key, value)  # attn_output: (N, B, D)
            attn_output = attn_output.transpose(0, 1)  # (B, N, D)

            # 融合车辆特征和注意力结果
            rnn_input = vehicle_features + attn_output  # (B, N, D)

            # 输入到 RNN
            rnn_input = rnn_input.view(B * N, 1, D)  # (B*N, 1, D)
            rnn_output, (h, c) = self.rnns[mode](rnn_input, (h, c))  # rnn_output: (B*N, 1, hidden_dim)

            # 预测当前轨迹
            pred = self.fcs[mode](rnn_output.view(B, N, self.hidden_dim))  # (B, N, future_steps * output_dim)
            pred = pred.view(B, N, self.future_steps, self.output_dim)  # (B, N, future_steps, output_dim)
            predictions.append(pred)

            # 保存当前解码器的隐藏状态
            hidden_states.append(rnn_output.view(B, N, self.hidden_dim))  # (B, N, hidden_dim)

        # 将所有预测结果拼接
        predictions = torch.stack(predictions, dim=2)  # (B, N, num_modes, future_steps, output_dim)

        # 将所有隐藏状态拼接
        hidden_states = torch.cat(hidden_states, dim=-1)  # (B, N, hidden_dim * num_modes)

        # 计算轨迹概率值
        mode_probs = self.prob_fc(hidden_states)  # (B, N, num_modes)
        mode_probs = F.softmax(mode_probs, dim=-1)  # 使用 softmax 归一化

        return predictions, mode_probs



class GRUDecoder(nn.Module):

    def __init__(self,
                 local_channels: int,
                 global_channels: int,
                 future_steps: int,
                 num_modes: int,
                 uncertain: bool = True,
                 min_scale: float = 1e-3) -> None:
        super(GRUDecoder, self).__init__()
        self.input_size = global_channels
        self.hidden_size = local_channels
        self.future_steps = future_steps
        self.num_modes = num_modes
        self.uncertain = uncertain
        self.min_scale = min_scale

        self.gru = nn.GRU(input_size=self.input_size,
                          hidden_size=self.hidden_size,
                          num_layers=1,
                          bias=True,
                          batch_first=False,
                          dropout=0,
                          bidirectional=False)
        self.loc = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_size, 2))
        if uncertain:
            self.scale = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.LayerNorm(self.hidden_size),
                nn.ReLU(inplace=True),
                nn.Linear(self.hidden_size, 2))
        self.pi = nn.Sequential(
            nn.Linear(self.hidden_size + self.input_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_size, 1))
        self.apply(init_weights)

    def forward(self,
                local_embed: torch.Tensor,
                global_embed: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B = local_embed.shape[0]
        N = local_embed.shape[1]
        D = local_embed.shape[2]
        # (B,N,6)
        pi = self.pi(torch.cat((local_embed.unsqueeze(1).expand(B, 6, N, D),
                                global_embed), dim=-1)).squeeze(-1).transpose(1,2)
        global_embed = global_embed.reshape(-1, self.input_size)  # [F x N, D]
        global_embed = global_embed.expand(self.future_steps, *global_embed.shape)  # [H, F x N, D]
        local_embed = local_embed.unsqueeze(2).repeat(1, 1, 6, 1) # [1, F x N, D]
        local_embed = local_embed.reshape(B * N * 6, D).unsqueeze(0)
        out, _ = self.gru(global_embed, local_embed)
        out = out.transpose(0, 1)  # [F x N, H, D]
        loc = self.loc(out)  # [F x N, H, 2]
        if self.uncertain:
            scale = F.elu_(self.scale(out), alpha=1.0) + 1.0 + self.min_scale  # [F x N, H, 2]
            return torch.cat((loc, scale),
                             dim=-1).view(self.num_modes, -1, self.future_steps, 4), pi  # [F, N, H, 4], [N, F]
        else:
            return loc.view(B, N, self.num_modes, self.future_steps, 2), pi  # [F, N, H, 2], [N, F]



class FusionNet2(nn.Module):
    def __init__(self, device, config):
        super(FusionNet2, self).__init__()
        self.device = device
        d_embed = config['d_embed']  # 128
        dropout = config['dropout']
        # True
        update_edge = config['update_edge']  # True

        self.proj_actor = nn.Sequential(
            # 128，128
            nn.Linear(config['d_actor'], d_embed),
            nn.LayerNorm(d_embed),
            nn.ReLU(inplace=True)
        )
        self.proj_lane = nn.Sequential(
            # 128，128
            nn.Linear(config['d_lane'], d_embed),
            nn.LayerNorm(d_embed),
            nn.ReLU(inplace=True)
        )
        self.proj_rpe_scene = nn.Sequential(
            # 5，128
            nn.Linear(5, config['d_rpe']),
            nn.LayerNorm(config['d_rpe']),
            nn.ReLU(inplace=True)
        )

        self.fuse_scene = SymmetricFusionTransformer(self.device,
                                                     # 128，128，8，4
                                                     d_model=d_embed,
                                                     d_edge=config['d_rpe'],
                                                     n_head=config['n_scene_head'],
                                                     n_layer=config['n_scene_layer'],
                                                     dropout=dropout,
                                                     update_edge=update_edge)

        self.sc_pass = SC_Pass(node_dim=d_embed,
                               edge_dim=d_embed,
                               embed_dim=d_embed,
                               num_heads=8,
                               dropout=dropout)

        self.edge_drop = DistanceDropEdge(100.0)

    def forward(self,
                actors: Tensor,
                actors_sc: Tensor,
                actor_idcs: List[Tensor],
                lanes: Tensor,
                lanes_sc: Tensor,
                lane_idcs: List[Tensor],
                rpe_prep: Dict[str, Tensor],
                rpe_sc: Dict[str, Tensor]):
        # 传参：车辆编码，对应车辆id，车道编码，对应车道id，相对位置RPE
        # print('actors: ', actors.shape)
        # print('actor_idcs: ', [x.shape for x in actor_idcs])
        # print('lanes: ', lanes.shape)
        # print('lane_idcs: ', [x.shape for x in lane_idcs])

        # projection
        # 车辆与车道编码进行维度对齐
        actors = self.proj_actor(actors)
        actors_sc = self.proj_actor(actors_sc)
        lanes = self.proj_lane(lanes)
        lanes_sc = self.proj_lane(lanes_sc)

        # 初始化更新列表
        actors_new, lanes_new, actors_sc_new, lanes_sc_new = list(), list(), list(), list()
        # 对每个批次中的信息进行处理
        for a_idcs, l_idcs, rpes, rpe_sc in zip(actor_idcs, lane_idcs, rpe_prep, rpe_sc):
            # 在一个batch内，各场景文件中的车辆id，车道id，相对边关系
            # * fusion - scene
            # 根据对应id找到该车辆/车道信息的编码---（N，128），(M, 128)
            _actors = actors[a_idcs]
            _lanes = lanes[l_idcs]
            # 根据索引取出车辆和车道特征，以0维纵向拼接（N+M, 128）
            tokens = torch.cat([_actors, _lanes], dim=0)  # (N+M, d_model)
            # 第一次转系坐标系下，元素间相对关系RPE--->（N+M，N+M，7）
            rpe = self.proj_rpe_scene(rpes['scene'].permute(1, 2, 0))  # (N+M, N+M, d_rpe)
            #print(rpe.shape)
            # 返回锚姿态更新后的节点特征
            # token中的一对节点，通过rpe更新其节点特征, mask在预处理时候没有传入值---返回更新后的节点特征编码（N，128）
            # 将局部坐标系进行注意力更新的边缘信息拿出
            out, _ = self.fuse_scene(tokens, rpe, rpes['scene_mask'])
            actors_new.append(out[:len(a_idcs)])
            lanes_new.append(out[len(a_idcs):])
            #print(out.shape)


            # 全局特征编码
            # 相对关系编码，rep(N+M, N+M, 2)---(N+M, N+M, 128)，N为车辆数和车道数的总数
            rpe2 = rpe_sc['scene']  # (N+M, N+M, 2)
            # 相对关系编码，rep---(N+M, N+M, 128)，N为车辆数和车道数的总数
            # 双坐标系拼接
            #print(rpe2.shape)
            #print(mem.shape)
            rpe_fusion = torch.cat([rpe2, rpe], dim=-1)
            # (N+M, N+M, 7)
            # print(rpe_fusion.shape)

            _actors_sc = actors_sc[a_idcs]
            _lanes_sc = lanes_sc[l_idcs]
            # 根据索引取出车辆和车道特征，以0维纵向拼接（N+M, 128）
            tokens_sc = torch.cat([_actors_sc, _lanes_sc], dim=0)  # (N+M, d_model)
            # 构建边索引矩阵
            # 构建边的连接关系 (2, E)
            N = _actors_sc.shape[0]
            M = _lanes_sc.shape[0]
            # print(N,M)
            # 边索引
            edge_index = rpe_fusion  # (N+M, N+M, 2+128)
            # print(edge_index.shape)
            rows, cols = torch.where(torch.ones(N + M, N + M))  # 全连接
            edge_index_connections = torch.stack([rows, cols], dim=0)  # (2, E)
            # print(edge_index_connections.shape)
            edge_attr = edge_index[rows, cols]  # (边数, 7)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # print(edge_attr.shape)
            # 将超出部分的边进行筛选
            edge_index_connections = edge_index_connections.to(device)
            edge_attr = edge_attr.to(device)
            edge_index, edge_attr = self.edge_drop(edge_index_connections, edge_attr)
            #print(edge_index.shape)
            #print(edge_attr.shape)
            out_sc = self.sc_pass(tokens_sc, edge_index, edge_attr)
            # 返回锚姿态更新后的节点特征
            # token中的一对节点，通过rpe更新其节点特征, mask在预处理时候没有传入值---返回更新后的节点特征编码（N，128）
            # 将局部坐标系进行注意力更新的边缘信息拿出
            # out, edge = self.fuse_scene(tokens, rpe, rpes['scene_mask'])

            # 前一部分是车辆特征，后一部分是车道特征，分别赋值给车辆与车道，作为节点更新操作
            actors_sc_new.append(out_sc[:len(a_idcs)])
            lanes_sc_new.append(out_sc[len(a_idcs):])

        # print('actors: ', [x.shape for x in actors_new])
        # print('lanes: ', [x.shape for x in lanes_new])

        # 分别拼接每个批次中的各场景更新后的车辆节点和车道节点特征
        actors = torch.cat(actors_new, dim=0)
        lanes = torch.cat(lanes_new, dim=0)
        actors_sc = torch.cat(actors_sc_new, dim=0)
        lanes_sc = torch.cat(lanes_sc_new, dim=0)
        # print('actors: ', actors.shape)
        # print('lanes: ', lanes.shape)

        # 返回SIMPL通过SFT更新后的车辆与车道编码，simpl直接在本部分完成后进行预测，GUP会有后续处理
        return actors, lanes, actors_sc, lanes_sc


from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj
from torch_geometric.typing import OptTensor
from torch_geometric.typing import Size
from torch_geometric.utils import softmax
from torch_geometric.utils import subgraph

class SC_Pass(MessagePassing):
    def __init__(self,
                 node_dim: int,
                 edge_dim: int,
                 embed_dim: int,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 **kwargs) -> None:
        super(SC_Pass, self).__init__(aggr='add', node_dim=0, **kwargs)
        self.embed_dim = embed_dim
        self.edge_dim = edge_dim
        self.num_heads = num_heads
        self.center_embed = SingleInputEmbedding(in_channel=node_dim, out_channel=embed_dim)
        self.nbr_embed = MultipleInputEmbedding(in_channels=[128, 128], out_channel=embed_dim)
        self.lin_q = nn.Linear(embed_dim, embed_dim)
        self.lin_k = nn.Linear(embed_dim, embed_dim)
        self.lin_v = nn.Linear(embed_dim, embed_dim)
        self.lin_self = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(dropout)
        self.lin_ih = nn.Linear(embed_dim, embed_dim)
        self.lin_hh = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout))
        self.apply(init_weights)

    def forward(self,
                x: torch.Tensor,
                edge_index: Adj,
                edge_attr: torch.Tensor,
                size: Size = None) -> torch.Tensor:
        # -------------------------------------------此步不执行，直接跳转-----------------------------------------------
        center_embed = self.center_embed(x)

            # 进行消息传递操作
        center_embed = center_embed + self._mha_block(self.norm1(center_embed), x, edge_index, edge_attr, size)
        center_embed = center_embed + self._ff_block(self.norm2(center_embed))
        return center_embed

    def message(self,
                edge_index: Adj,
                center_embed_i: torch.Tensor,
                x_j: torch.Tensor,
                edge_attr: torch.Tensor,
                index: torch.Tensor,
                ptr: OptTensor,
                size_i: Optional[int]) -> torch.Tensor:
        # 聚合边和源节点的编码
        nbr_embed = self.nbr_embed([x_j, edge_attr])
        query = self.lin_q(center_embed_i).view(-1, self.num_heads, self.embed_dim // self.num_heads)
        key = self.lin_k(nbr_embed).view(-1, self.num_heads, self.embed_dim // self.num_heads)
        value = self.lin_v(nbr_embed).view(-1, self.num_heads, self.embed_dim // self.num_heads)
        scale = (self.embed_dim // self.num_heads) ** 0.5
        alpha = (query * key).sum(dim=-1) / scale
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = self.attn_drop(alpha)
        return value * alpha.unsqueeze(-1)

    def update(self,
               inputs: torch.Tensor,
               center_embed: torch.Tensor) -> torch.Tensor:
        inputs = inputs.view(-1, self.embed_dim)
        gate = torch.sigmoid(self.lin_ih(inputs) + self.lin_hh(center_embed))
        return inputs + gate * (self.lin_self(center_embed) - inputs)

    def _mha_block(self,
                   center_embed: torch.Tensor,
                   x: torch.Tensor,
                   edge_index: Adj,
                   edge_attr: torch.Tensor,
                   size: Size) -> torch.Tensor:
        center_embed = self.out_proj(self.propagate(edge_index=edge_index, x=x, center_embed=center_embed,
                                                    edge_attr=edge_attr, size=size))
        return self.proj_drop(center_embed)

    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)



class SingleInputEmbedding(nn.Module):

    def __init__(self,
                 in_channel: int,
                 out_channel: int) -> None:
        super(SingleInputEmbedding, self).__init__()
        self.embed = nn.Sequential(
            nn.Linear(in_channel, out_channel),
            nn.LayerNorm(out_channel),
            nn.ReLU(inplace=True),
            nn.Linear(out_channel, out_channel),
            nn.LayerNorm(out_channel),
            nn.ReLU(inplace=True),
            nn.Linear(out_channel, out_channel),
            nn.LayerNorm(out_channel))
        self.apply(init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embed(x)


class MultipleInputEmbedding(nn.Module):

    def __init__(self,
                 in_channels: List[int],
                 out_channel: int) -> None:
        super(MultipleInputEmbedding, self).__init__()
        self.module_list = nn.ModuleList(
            [nn.Sequential(nn.Linear(in_channel, out_channel),
                           nn.LayerNorm(out_channel),
                           nn.ReLU(inplace=True),
                           nn.Linear(out_channel, out_channel))
             for in_channel in in_channels])
        self.aggr_embed = nn.Sequential(
            nn.LayerNorm(out_channel),
            nn.ReLU(inplace=True),
            nn.Linear(out_channel, out_channel),
            nn.LayerNorm(out_channel))
        self.apply(init_weights)

    def forward(self,
                continuous_inputs: List[torch.Tensor],
                categorical_inputs: Optional[List[torch.Tensor]] = None) -> torch.Tensor:
        for i in range(len(self.module_list)):
            continuous_inputs[i] = self.module_list[i](continuous_inputs[i])
        output = torch.stack(continuous_inputs).sum(dim=0)
        if categorical_inputs is not None:
            output += torch.stack(categorical_inputs).sum(dim=0)
        return self.aggr_embed(output)


class DistanceDropEdge:
    def __init__(self, max_distance: Optional[float] = None) -> None:
        self.max_distance = max_distance

    def __call__(self,
                 edge_index: torch.Tensor,
                 edge_attr: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # 如果未设置最大距离，直接返回原始数据
        if self.max_distance is None:
            return edge_index, edge_attr

        # 确保 edge_index 和 edge_attr 在同一设备上
        if edge_index.device != edge_attr.device:
            raise RuntimeError(
                f"edge_index and edge_attr must be on the same device, "
                f"but got edge_index on {edge_index.device} and edge_attr on {edge_attr.device}"
            )
        #print(edge_attr[:, :2].shape)
        # 拿出前两列的向量
        vec = edge_attr[:, :2]
        # 计算 edge_attr 的 L2 范数（欧几里得距离）
        mask = torch.norm(vec, p=2, dim=-1) < self.max_distance

        # 筛选 edge_index 和 edge_attr
        row, col = edge_index
        edge_index = torch.stack([row[mask], col[mask]], dim=0)
        edge_attr = edge_attr[:, 2:]
        edge_attr = edge_attr[mask]

        return edge_index, edge_attr

