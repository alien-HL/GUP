import os
import math
import random
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from utils.utils import from_numpy


class ArgoDataset(Dataset):
    # 自动调用，初始化方法
    def __init__(self,
                 dataset_dir: str,
                 mode: str,
                 obs_len: int = 20,
                 pred_len: int = 30,
                 aug: bool = False,
                 verbose: bool = False):
        self.mode = mode
        self.aug = aug
        self.verbose = verbose

        self.dataset_files = []
        self.dataset_len = -1
        self.prepare_dataset(dataset_dir)

        self.obs_len = obs_len
        self.pred_len = pred_len
        self.seq_len = obs_len + pred_len

        if self.verbose:
            print('[Dataset] Dataset Info:')
            print('-- mode: ', self.mode)
            print('-- total frames: ', self.dataset_len)
            print('-- obs_len: ', self.obs_len)
            print('-- pred_len: ', self.pred_len)
            print('-- seq_len: ', self.seq_len)
            print('-- aug: ', self.aug)

    def prepare_dataset(self, feat_path):
        if self.verbose:
            print("[Dataset] preparing {}".format(feat_path))

        if isinstance(feat_path, list):
            for path in feat_path:
                sequences = os.listdir(path)
                sequences = sorted(sequences)
                for seq in sequences:
                    file_path = f"{path}/{seq}"
                    self.dataset_files.append(file_path)
        else:
            sequences = os.listdir(feat_path)
            sequences = sorted(sequences)
            for seq in sequences:
                file_path = f"{feat_path}/{seq}"
                self.dataset_files.append(file_path)

        self.dataset_len = len(self.dataset_files)

    # 自动调用，数据集长度
    def __len__(self):
        return self.dataset_len

    # 自动调用，按照索引读取数据集，获取信息
    def __getitem__(self, idx):
        # 按索引读预处理数据文件
        df = pd.read_pickle(self.dataset_files[idx])
        '''
            "SEQ_ID", "CITY_NAME",
            "ORIG", "ROT",
            "TIMESTAMP", "TRAJS", "TRAJS_CTRS", "TRAJS_VECS", "TRAJS_VECTORS",
            "LANE_GRAPH"
        '''

        # 是否对数据进行增强操作
        data = self.data_augmentation(df)

        seq_id = data['SEQ_ID']
        city_name = data['CITY_NAME']
        orig = data['ORIG']
        rot = data['ROT']

        # 二次转系各车辆轨迹（各车辆19局部坐标系）
        trajs = data['TRAJS']
        # 一次转系各车辆轨迹（Agent19全局坐标系）
        trajs_ori = data['TRAJS_ORI']

        # 二次转系各车辆坐标进行分割（历史+未来）
        trajs_obs = trajs[:, :self.obs_len]
        trajs_fut = trajs[:, self.obs_len:]

        # 二次转系各车辆轨迹向量分割
        trajs_vector = data['TRAJS_VECTORS']
        # 二次转系各车辆坐标进行分割（历史+未来）
        trajs_vector_obs = trajs_vector[:, :self.obs_len]
        trajs_vector_fut = trajs_vector[:, self.obs_len:]

        # 一次转系各车辆坐标进行分割（历史+未来）
        trajs_obs_ori = trajs_ori[:, :self.obs_len]
        trajs_fut_ori = trajs_ori[:, self.obs_len:]

        # 缺值填充矩阵分割（历史+未来）
        pad_flags = data['PAD_FLAGS']
        pad_obs = pad_flags[:, :self.obs_len]
        pad_fut = pad_flags[:, self.obs_len:]

        # 车辆锚姿态：一次转系各车19步（二次转系中心），各车0->19向量姿态[cos,sin]
        trajs_ctrs = data['TRAJS_CTRS']
        trajs_vecs = data['TRAJS_VECS']

        # 车道特征
        graph = data['LANE_GRAPH']

        '''
            'node_ctrs'         (164, 10, 2)
            'node_vecs'         (164, 10, 2)
            'sc_vecs'           (164, 10, 2) 
            'turn'              (164, 10, 2)
            'control'           (164, 10)
            'intersect'         (164, 10)
            'left'              (164, 10)
            'right'             (164, 10)
            'lane_ctrs'         (164, 2)
            'lane_vecs'         (164, 2)
            'num_nodes'         1640
            'num_lanes'         164
            # 超出距离元素舍弃
            'rel_lane_flags'    (164,)
        '''

        # 一次转系后的细分车道总中心点，细分车道段总单位向量表征（（首->尾）/长度）
        lane_ctrs = graph['lane_ctrs']
        lane_vecs = graph['lane_vecs']

        # 元素之间相对关系列表
        rpes = dict()
        rpes_sc = dict()
        # 纵向拼接：一次转系后的各车19坐标 + 一次转系后的车道中心坐标 ---（num_agent+num_lane，2）
        scene_ctrs = torch.cat([torch.from_numpy(trajs_ctrs), torch.from_numpy(lane_ctrs)], dim=0)
        # 纵向拼接：一次转系后的各车0->19向量姿态[cos,sin](num_agent, 2) + 一次转系后的车道单位向量（num_lane, 2）---（num_agent+num_lane，2）
        scene_vecs = torch.cat([torch.from_numpy(trajs_vecs), torch.from_numpy(lane_vecs)], dim=0)

        # 第一次转系坐标系下，元素间相对关系RPE计算---（5，N+M，N+M）
        rpes['scene'], rpes['scene_mask'] = self._get_rpe(scene_ctrs, scene_vecs)
        # 相对关系为2维向量信息
        rpes_sc['scene'], rpes_sc['scene_mask'] = self._get_rpe_sc(scene_ctrs, scene_vecs)

        data = {}
        data['SEQ_ID'] = seq_id
        data['CITY_NAME'] = city_name
        data['ORIG'] = orig
        data['ROT'] = rot
        data['TRAJS_OBS'] = trajs_obs
        data['TRAJS_FUT'] = trajs_fut
        data['TRAJS_OBS_ORI'] = trajs_obs_ori
        data['TRAJS_FUT_ORI'] = trajs_fut_ori
        data['TRAJS_VECTORS_OBS'] = trajs_vector_obs
        data['TRAJS_VECTORS_FUT'] = trajs_vector_fut
        data['PAD_OBS'] = pad_obs
        data['PAD_FUT'] = pad_fut
        data['TRAJS_CTRS'] = trajs_ctrs
        data['TRAJS_VECS'] = trajs_vecs
        data['LANE_GRAPH'] = graph
        data['RPE'] = rpes
        data['RPE_SC'] = rpes_sc

        return data

    def _get_cos(self, v1, v2):
        v1_norm = v1.norm(dim=-1)
        v2_norm = v2.norm(dim=-1)
        v1_x, v1_y = v1[..., 0], v1[..., 1]
        v2_x, v2_y = v2[..., 0], v2[..., 1]
        cos_angle = (v1_x * v2_x + v1_y * v2_y) / (v1_norm * v2_norm + 1e-10)
        return cos_angle

    def _get_sin(self, v1, v2):
        v1_norm = v1.norm(dim=-1)
        v2_norm = v2.norm(dim=-1)
        v1_x, v1_y = v1[..., 0], v1[..., 1]
        v2_x, v2_y = v2[..., 0], v2[..., 1]
        sin_angle = (v1_x * v2_y - v1_y * v2_x) / (v1_norm * v2_norm + 1e-10)
        return sin_angle

    def _get_rpe(self, ctrs, vecs, radius=100.0):
        # 计算两个元素间的相对距离：传参维度都为（num_agent+num_lane，2），为简洁我们表示为N与M
        # （1，N+M，2）- （N+M，1，2）= （N+M，N+M, 2）,沿-1轴计算欧式距离---（N+M, N+M, 1）
        d_pos = (ctrs.unsqueeze(0) - ctrs.unsqueeze(1)).norm(dim=-1)
        # 归一化相对距离到一定范围内
        d_pos = d_pos * 2 / radius
        # vec = ctrs.unsqueeze(0) - ctrs.unsqueeze(1)
        # 增加一维（1，N+M，N+M，2）
        pos_rpe = d_pos.unsqueeze(0)

        # 拆分最后一个维度
        # split_tensors = torch.split(pos_rpe, 1, dim=-1)  # 得到两个 (1, 5, 5, 1) 的张量
        # 拼接第一个维度
        # pos_rpe = torch.cat(split_tensors, dim=0)  # 得到 (2, 5, 5, 1)

        # 定义方法。计算两个向量之间的正余弦值
        def compute_cos_sin(vec1, vec2):
            cos_ag = self._get_cos(vec1, vec2)
            sin_ag = self._get_sin(vec1, vec2)
            return cos_ag, sin_ag

        # 与距离计算一致，得到（N+M, N+M，1）
        cos_ag1, sin_ag1 = compute_cos_sin(vecs.unsqueeze(0), vecs.unsqueeze(1))

        v_pos = ctrs.unsqueeze(0) - ctrs.unsqueeze(1)
        # （N+M, N+M，1）
        cos_ag2, sin_ag2 = compute_cos_sin(vecs.unsqueeze(0), v_pos)

        # 角度拼接, 默认沿着第0维拼接---（4，N+M，N+M，1）
        ang_rpe = torch.stack([cos_ag1, sin_ag1, cos_ag2, sin_ag2])
        # 拼接距离信息，不增加新维度，（5，N+M，N+M，1），最后1维可以省略---（5，N+M，N+M）
        rpe = torch.cat([ang_rpe, pos_rpe], dim=0)

        return rpe, None

    def _get_rpe_sc(self, ctrs, vecs, radius=100.0):
        # 计算两个元素间的相对距离：传参维度都为（num_agent+num_lane，2），为简洁我们表示为N与M
        # （1，N+M，2）- （N+M，1，2）= （N+M，N+M, 2）,沿-1轴计算欧式距离---（N+M, N+M, 1）
        vec = ctrs.unsqueeze(0) - ctrs.unsqueeze(1)
        return vec, None

    # 运行完初始化，__get__等自调用函数后，重写pytorch.dataloader内置函数
    #                DataLoader(train_set,
    #                           batch_size=args.train_batch_size,
    #                           shuffle=True,
    #                           num_workers=0,#default = 8
    #                           collate_fn=train_set.collate_fn,
    #                           drop_last=True,
    #                           pin_memory=False)
    # 按照collate_fn中的要求，对一个batch的预处理后的数据进行整理，多用于输入网络模型前

    def collate_fn(self, batch: List[Any]) -> Dict[str, Any]:
        # 传参：batch
        batch = from_numpy(batch)
        # 字典型数据初始化
        data = dict()
        data['BATCH_SIZE'] = len(batch)

        # 重新打包以有的键值数据，并加入新的键值
        for key in batch[0].keys():
            data[key] = [x[key] for x in batch]
        '''
            Keys:
            'BATCH_SIZE', 'SEQ_ID', 'CITY_NAME',
            'ORIG', 'ROT',
            'TRAJS_OBS', 'PAD_OBS', 'TRAJS_FUT', 'PAD_FUT', 'TRAJS_CTRS', 'TRAJS_VECS', 'TRAJS_OBS_ORI' , 'TRAJS_FUT_ORI'
            'LANE_GRAPH',
            'RPE'
        '''

        # 传参：批次数b，该批次二次转系下的各车辆历史坐标（b, N，20，2），该批次下历史缺值填充矩阵（b, N，20）--- 返回这一个批次内的所有车辆轨迹（N_b，3，20），以及车辆索引列表（N_b）
        actors, actor_idcs = self.actor_gather(data['BATCH_SIZE'], data['TRAJS_OBS'], data['PAD_OBS'])
        # 传参：批次数b，该批次一次转系下的各车辆历史坐标（b, N，20，2），该批次下历史缺值填充矩阵（b, N，20）--- 返回这一个批次内的所有车辆轨迹（N_b，3，20）
        actors_sc, _ = self.actor_gather_sc(data['BATCH_SIZE'], data['TRAJS_OBS_ORI'], data['PAD_OBS'])
        # 传参：批次数b，车道图（b, ）--- 返回一个批次的车道二次转系信息（M_b, 10, 10），车道索引列表（M_b），车道一次转系信息（M_b, 10, 8）
        lanes, lane_idcs, lanes_sc = self.graph_gather(data['BATCH_SIZE'], data["LANE_GRAPH"])

        # 赋值给字典型数据，输入模型
        data['ACTORS'] = actors
        data['ACTORS_SC'] = actors_sc
        data['ACTOR_IDCS'] = actor_idcs
        data['LANES'] = lanes
        data['LANES_SC'] = lanes_sc
        data['LANE_IDCS'] = lane_idcs
        return data

    def actor_gather(self, batch_size, actors, pad_flags):
        # 传参：批次数，二次转系下的各车辆历史坐标（b, N，20，2），历史缺值填充矩阵（b, N，20）
        # 每个批次下的车辆个数: [N1, N2, ...]
        num_actors = [len(x) for x in actors]

        # 车辆索引矩阵初始化
        actor_idcs = []
        act_feats = []
        count = 0
        for i in range(batch_size):
            # 按批次循环，拼接（N，20，2）+ （N，20，1）= （N，20，3）
            act_feats.append(torch.cat([actors[i], pad_flags[i].unsqueeze(2)], dim=2))
        # 将所有批次都作如此拼接，且将每个批次的1，2维度交换：（N，3，20），得到act_feats
        act_feats = [x.transpose(1, 2) for x in act_feats]
        # [N_a, 3, 20], N_a is agent number in a batch
        # 将列表按照第0维拼接---（N，3，20）
        actors = torch.cat(act_feats, 0)

        for i in range(batch_size):
            # 按批次循环，赋值每个批次下的车辆id，且id不重复，下一个批次顺延
            idcs = torch.arange(count, count + num_actors[i])
            actor_idcs.append(idcs)
            count += num_actors[i]
        # 返回这一个批次内的所有车辆轨迹（N_b，3，20），以及车辆索引列表（N_b）
        return actors, actor_idcs

    def actor_gather_sc(self, batch_size, actors, pad_flags):
        num_actors = [len(x) for x in actors]
        idx_actor = []
        features_act = []
        count = 0

        for i in range(batch_size):
            features_act.append(torch.cat([actors[i], pad_flags[i].unsqueeze(2)], dim=2))
        features_act = [x.transpose(1, 2) for x in features_act]
        actors = torch.cat(features_act, 0)

        for i in range(batch_size):
            idx = torch.arange(count, count + num_actors[i])
            idx_actor.append(idx)
            count += num_actors[i]

        return actors, idx_actor

    def graph_gather(self, batch_size, graphs):

        # 车道索引列表初始化
        lane_idx = []
        lane_count = 0

        for i in range(batch_size):
            # 在一个批次中循环，拿出该批次中，一个文件的场景信息，拼接其车道索引矩阵和车道数量
            l_idcs = torch.arange(lane_count, lane_count + graphs[i]["num_lanes"])
            lane_idx.append(l_idcs)
            lane_count += graphs[i]["num_lanes"]

        # 拼接字典：二次转系后的车道信息
        graph = {key: torch.cat([x[key] for x in graphs], 0) for key in
                 ["node_ctrs", "node_vecs", "turn", "control", "intersect", "left", "right"]}
        # 拼接字典：一次转系后的车道中线坐标
        graph_sc = {key: torch.cat([x[key] for x in graphs], 0) for key in
                    ["sc_vecs", "turn", "control", "intersect", "left", "right"]}
        # 更新字典：一次转系后的车道锚姿态中心和单位向量
        graph.update({key: [x[key] for x in graphs] for key in ["lane_ctrs", "lane_vecs"]})

        '''
                    'node_ctrs'         (164, 10, 2)
                    'node_vecs'         (164, 10, 2)
                    'sc_vecs'           (164, 10, 2) 
                    'turn'              (164, 10, 2)
                    'control'           (164, 10)
                    'intersect'         (164, 10)
                    'left'              (164, 10)
                    'right'             (164, 10)
                    'lane_ctrs'         (164, 2)
                    'lane_vecs'         (164, 2)
                    'num_nodes'         1640
                    'num_lanes'         164
                    # 超出距离元素舍弃
                    'rel_lane_flags'    (164,)
        '''
        # 二次转系，沿最后一维拼接车道信息：（M，10，10）
        lanes = torch.cat([
            graph['node_ctrs'],
            graph['node_vecs'],
            graph['turn'],
            graph['control'].unsqueeze(2),
            graph['intersect'].unsqueeze(2),
            graph['left'].unsqueeze(2),
            graph['right'].unsqueeze(2)
        ], dim=-1)

        # 一次转系，沿最后一维拼接车道信息：（M，10，8）
        lanes_sc = torch.cat([
            graph_sc['sc_vecs'],
            graph_sc['turn'],
            graph_sc['control'].unsqueeze(2),
            graph_sc['intersect'].unsqueeze(2),
            graph_sc['left'].unsqueeze(2),
            graph_sc['right'].unsqueeze(2)
        ], dim=-1)

        # 返回一个批次的车道二次转系信息，车道索引列表，车道一次转系信息
        return lanes, lane_idx, lanes_sc

    def rpe_gather(self, rpes):
        rpe = dict()
        for key in list(rpes[0].keys()):
            rpe[key] = [x[key] for x in rpes]
        return rpe

    def data_augmentation(self, df):

        data = {key: df[key].values[0] for key in df.keys()}

        is_aug = random.choices([True, False], weights=[0.3, 0.7])[0]
        if not (self.aug and is_aug):
            return data

        def flip_vertical(coords):
            coords[..., 1] *= -1

        flip_vertical(data['TRAJS_CTRS'])
        flip_vertical(data['TRAJS_VECS'])
        flip_vertical(data['TRAJS'])

        lane_graph_keys = ['lane_ctrs', 'lane_vecs', 'node_ctrs', 'node_vecs']
        for key in lane_graph_keys:
            flip_vertical(data['LANE_GRAPH'][key])

        data['LANE_GRAPH']['left'], data['LANE_GRAPH']['right'] = (
            data['LANE_GRAPH']['right'], data['LANE_GRAPH']['left']
        )

        return data