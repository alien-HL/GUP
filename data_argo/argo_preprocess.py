import os
from os.path import expanduser
import time
from typing import Any, Dict, List, Tuple
import random
import copy
import numpy as np
import pandas as pd
import pickle as pkl
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.path import Path
from shapely.geometry import LineString, Point, Polygon, MultiPolygon
from shapely.ops import cascaded_union, nearest_points, unary_union
from scipy import sparse, spatial
from argoverse.map_representation.map_api import ArgoverseMap
import init_path



class ArgoPreproc():
    def __init__(self, args, verbose=False):
        self.args = args
        self.verbose = verbose
        self.debug = args.debug
        self.viz = args.viz
        self.mode = args.mode

        self.MAP_RADIUS = 80.0
        self.COMPL_RANGE = 30.0
        # 相关道路
        self.REL_LANE_THRES = 30.0

        # 细分车道段
        self.SEG_LENGTH = 10.0
        self.SEG_N_NODE = 10

        self.argo_map = ArgoverseMap()

        if self.debug:
            self.map_vis = ArgoMapVisualizer()

    def print(self, info):
        if self.verbose:
            print(info)

    def process(self, seq_id, df):
        # 取城市名
        city_name = df['CITY_NAME'].values[0]

        # -----------------------------------------------轨迹信息处理--------------------------------------------------

        # 拿到轨迹信息：ts时间戳，trajs_ori三部分[agent，av，others...]---（N，50，2），pad_flags[N，50...]各车无效步填充标志
        ts, trajs_ori, pad_flags = self.get_trajectories(df, city_name)

        # get origin and rot
        # 第一部分为agent所有轨迹信息，传参，求出以agent19为中心的坐标系变换
        orig, rot = self.get_origin_rotation(trajs_ori[0])  # * ego-centric
        # 第一次标准化步骤：原始坐标系——>Agent19坐标系
        trajs_ori = (trajs_ori - orig).dot(rot)

        # ~ normalize trajs  初始化轨迹坐标，二次标准化步骤：Agent19坐标系——>各车19坐标系
        trajs_norm = []  # 各车19坐标系所有轨迹信息
        trajs_ctrs = []  # 各车19步坐标（坐标系旋转中心）
        trajs_vecs = []  # 各车向量表示（cos，sin）

        # 第二次转系：将各车以自车19步的坐标，0->19向量的旋转矩阵进行转系操作
        for traj in trajs_ori:
            # 各车辆19步坐标点
            act_orig = traj[self.args.obs_len - 1]
            act_vec = act_orig - traj[0]
            theta = np.arctan2(act_vec[1], act_vec[0])
            act_rot = np.array([[np.cos(theta), -np.sin(theta)],
                                [np.sin(theta), np.cos(theta)]])
            # 追加列表操作
            trajs_norm.append((traj - act_orig).dot(act_rot))
            trajs_ctrs.append(act_orig)
            # 单位向量表征
            trajs_vecs.append(np.array([np.cos(theta), np.sin(theta)]))

        # numpy类型转换
        # 得到各车第二次转系后的归一化轨迹坐标，锚姿态旋转中心（各车19步坐标），锚姿态相对方向表征（cos，sin）
        trajs = np.array(trajs_norm)
        trajs_ctrs = np.array(trajs_ctrs)
        trajs_vecs = np.array(trajs_vecs)



        # -----------------------------------------------车道信息处理--------------------------------------------------

        # get ROI---以最感兴趣agent19时刻坐标，查询半径R=80范围内的车道id
        lane_ids = self.get_related_lanes(seq_id, city_name, orig, expand_dist=self.MAP_RADIUS)
        # get lane graph---得到车道信息，以graph字典返回
        lane_graph = self.get_lane_graph(seq_id, df, city_name, orig, rot, lane_ids)

        # find relevant lanes
        # 拿出agent19(全局坐标系)的所有车辆轨迹坐标，无差别拿出
        trajs_ori_flat = trajs_ori.reshape(-1, 2)
        # 拿出agent19(全局坐标系)的车道中心坐标（锚姿态）
        lane_ctrs = lane_graph['lane_ctrs']

        # 库函数：spatial.distance.cdist，在二维数据点中求欧式距离
        dists = spatial.distance.cdist(trajs_ori_flat, lane_ctrs)  # calculate distance
        # 满足条件的车道：轨迹点-车道中心距离 < 半径范围（30m）
        rel_lane_flags = np.min(dists, axis=0) < self.REL_LANE_THRES  # find lanes that are not close to the trajectory

        # 追加字典：符合R内的车道节点标志，筛选车辆周围30m的地图
        lane_graph['rel_lane_flags'] = rel_lane_flags

        # collect data
        # 与SIMPL不同的是agent19全局坐标下的轨迹坐标trajs_ori
        # 场景id，城市名，第一次转系agent19（原点，旋转矩阵），时间戳（各时间步距开始时间戳），第二次转系（各车19）后的各车坐标，一次转系后各车19步坐标（二次旋转中心），无效步填充标志，车道图，第一次转系各车坐标
        data = [[seq_id, city_name, orig, rot, ts, trajs, trajs_ctrs, trajs_vecs, pad_flags, lane_graph, trajs_ori]]
        headers = ["SEQ_ID", "CITY_NAME", "ORIG", "ROT", "TIMESTAMP",
                   "TRAJS", "TRAJS_CTRS", "TRAJS_VECS", "PAD_FLAGS", "LANE_GRAPH", "TRAJS_ORI"]

        # ! For debug
        if self.debug and self.viz:
            _, ax = plt.subplots(figsize=(10, 10))
            ax.axis('equal')
            vis_map = False
            self.plot_trajs(ax, trajs, trajs_ctrs, trajs_vecs, pad_flags, orig, rot, vis_map=vis_map)
            self.plot_lane_graph(ax, city_name, orig, rot, lane_ids, lane_graph, vis_map=vis_map)
            ax.set_title("{} {}".format(seq_id, city_name))
            plt.show()

        return data, headers

    def get_origin_rotation(self, traj):
        # 传参：agent所有轨迹信息
        # agent19步坐标
        orig = traj[self.args.obs_len - 1]
        # 第0步指向19步向量
        vec = orig - traj[0]
        # 求反正切函数，算向量夹角
        theta = np.arctan2(vec[1], vec[0])
        # agent0——>agent19，旋转矩阵表征，由且仅有一次转系操作
        # 注：无AV坐标系变换，直接将原始坐标转化为Agent坐标系。当轨迹测试后处理时，需要乘逆旋转矩阵，将Agent坐标系转化为原始地图坐标系。
        rot = np.array([[np.cos(theta), -np.sin(theta)],
                        [np.sin(theta), np.cos(theta)]])

        return orig, rot

    def get_trajectories(self,
                         df: pd.DataFrame,
                         city_name: str):
        # 时间戳unique后排序
        ts = np.sort(np.unique(df['TIMESTAMP'].values)).astype(float)
        # 拿出最后一步(19)时间戳
        t_obs = ts[self.args.obs_len - 1]

        # agent所有时间步(50)信息
        agent_traj = df[df["OBJECT_TYPE"] == "AGENT"]
        # 拼接x，y坐标，升维，（50，2），取前两列
        agent_traj = np.stack((agent_traj['X'].values, agent_traj['Y'].values), axis=1).astype(float)
        # 此代码无作用，只取轨迹前两列
        agent_traj[:, 0:2] = agent_traj[:, 0:2]

        # av所有时间步信息，同样操作，提取av轨迹
        av_traj = df[df["OBJECT_TYPE"] == "AV"]
        av_traj = np.stack((av_traj['X'].values, av_traj['Y'].values), axis=1).astype(float)
        av_traj[:, 0:2] = av_traj[:, 0:2]

        # 断言函数，条件满足无任何输出：由于数据集采集时，av与agent步数是全的
        assert len(agent_traj) == len(av_traj), "Shape error for AGENT and AV, AGENT: {}, AV: {}".format(
            agent_traj.shape, av_traj.shape)

        # 轨迹列表组成，把agent与av轨迹，依次拼接列表
        trajs = [agent_traj, av_traj]

        # agent最后一步(19)，未来预测都会以最后一步为中心进行坐标系旋转---预测中心
        pred_ctr = agent_traj[self.args.obs_len - 1]

        # 以时间戳一样的尺度进行初始化，作为agent和av无效坐标的填充标志
        pad_flags = [np.ones_like(ts), np.ones_like(ts)]

        # 拿出所有车辆id
        track_ids = np.unique(df["TRACK_ID"].values)
        # 遍历，拿出每辆对应id车辆（others）的轨迹信息，不包括agent和av
        for idx in track_ids:
            mot_traj = df[df["TRACK_ID"] == idx]
            # 若该车辆为agent或av，直接跳出循环
            if mot_traj['OBJECT_TYPE'].values[0] == 'AGENT' or mot_traj['OBJECT_TYPE'].values[0] == 'AV':
                continue

            # 其他车辆时间戳
            ts_mot = np.array(mot_traj['TIMESTAMP'].values).astype(float)
            mot_traj = np.stack((mot_traj['X'].values, mot_traj['Y'].values), axis=1).astype(float)

            # ~ remove traj after t_obs
            # 若该车辆的时间戳全部在agent19后，则视为该车前20步未出现，无需对其做处理，跳出循环，无需追加到后续轨迹列表中
            if np.all(ts_mot > t_obs):
                continue

            # 相交函数，第一个位置返回两者都含有的元素，第二个位置返回公共元素在ts的索引, 第三个位置返回交集在ts_mot的索引
            # 表征others车辆在哪几个时间步长有效
            _, idcs, _ = np.intersect1d(ts, ts_mot, return_indices=True)
            # 填充
            padded = np.zeros_like(ts)
            # 将无需填充的步，填充标志置1
            padded[idcs] = 1

            # 若该车第19步无值，则跳出循环，无需追加到后续轨迹列表中，只处理19步出现的车辆
            if not padded[self.args.obs_len - 1]:
                continue

            # np.full创建数组，形状与agent相同（50，2）
            mot_traj_pad = np.full(agent_traj[:, :2].shape, None)
            # 将有效时间步的车辆坐标通过索引进行赋值
            mot_traj_pad[idcs] = mot_traj

            mot_traj_pad = self.padding_traj_nn(mot_traj_pad)
            assert np.all(mot_traj_pad[idcs] == mot_traj), "Padding error"

            # 拼接others车辆的 x，y 坐标
            mot_traj = np.stack((mot_traj_pad[:, 0], mot_traj_pad[:, 1]), axis=1)
            mot_traj[:, 0:2] = mot_traj[:, 0:2]

            # 各车辆19步坐标
            mot_ctr = mot_traj[self.args.obs_len - 1]
            # 在19步，若该车距离与Agent相距太远的话，跳出循环，不对其预测，无需追加到后续轨迹列表中
            if np.linalg.norm(mot_ctr - pred_ctr) > self.MAP_RADIUS:
                continue

            # 向trajs追加列表--- [agent，av，others...]，形状---（N，50，2）
            trajs.append(mot_traj)
            # 向无效值填充矩阵追加列表--- [agent，av，others...]，形状---（N，50）
            pad_flags.append(padded)

        # 所有时间距离开始时间的距离---（50）
        ts = (ts - ts[0]).astype(np.float32)
        # 有效车辆坐标---（N，50，2）
        trajs = np.array(trajs).astype(np.float32)
        # 车辆无效步填充矩阵---（N，50）
        pad_flags = np.array(pad_flags).astype(np.int16)
        # 返回参数：距开始时间戳，轨迹信息，填充矩阵
        return ts, trajs, pad_flags

    def padding_traj_nn(self, traj):
        n = len(traj)

        # forward
        buff = None
        for i in range(n):
            if np.all(buff == None) and np.all(traj[i] != None):
                buff = traj[i]
            if np.all(buff != None) and np.all(traj[i] == None):
                traj[i] = buff
            if np.all(buff != None) and np.all(traj[i] != None):
                buff = traj[i]

        # backward
        buff = None
        for i in reversed(range(n)):
            if np.all(buff == None) and np.all(traj[i] != None):
                buff = traj[i]
            if np.all(buff != None) and np.all(traj[i] == None):
                traj[i] = buff
            if np.all(buff != None) and np.all(traj[i] != None):
                buff = traj[i]

        return traj

    def get_related_lanes(self, seq_id, city_name, orig, expand_dist):
        lane_ids = self.argo_map.get_lane_ids_in_xy_bbox(orig[0], orig[1], city_name, self.MAP_RADIUS)
        return copy.deepcopy(lane_ids)

    def get_lane_graph(self, seq_id, df, city_name, orig, rot, lane_ids):
        # sc_vecs表示全局坐标系（即第一次转系agent19）下的信息
        node_ctrs, node_vecs, turn, control, intersect, left, right, sc_vecs = [], [], [], [], [], [], [], []
        lane_ctrs, lane_vecs = [], []

        # 初始化地图api---取出该城市车道中心线字典
        lanes = self.argo_map.city_lane_centerlines_dict[city_name]

        for lane_id in lane_ids:
            lane = lanes[lane_id]
            # （原始坐标系）得到对应车道id的中心线坐标---（10，2）
            cl_sc = lane.centerline
            # LineString 将车道中心线坐标连接成一条线
            cl_ls = LineString(cl_sc)

            # np.floor向下取整
            # 将车道按10m一段进行分割。若不满10m，则就分一段
            num_segs = np.max([int(np.floor(cl_ls.length / self.SEG_LENGTH)), 1])

            # 计算每段长度
            ds = cl_ls.length / num_segs
            for i in range(num_segs):
                # 某段的起点与终点
                s_lb = i * ds
                s_ub = (i + 1) * ds
                # 在每段内均值插入10个点
                num_sub_segs = self.SEG_N_NODE

                # 全局坐标系下初始化列表
                cl_pts_sc = []
                # np.linspace 用于在指定的间隔内返回均匀间隔的数字。它非常适合用于生成指定范围内的连续值
                # np.linspace(s_lb, s_ub, num_sub_segs) 在 s_lb（下限）和 s_ub（上限）之间生成 num_sub_segs 均匀间隔的点。
                # 在一段细分的车道段内---均匀生成10个点
                for s in np.linspace(s_lb, s_ub, num_sub_segs):
                    cl_pts_sc.append(cl_ls.interpolate(s))

                # 在插入等间隔点之后，连接成线，可通过coords属性访问这些插值点的坐标值
                ctrln_sc = np.array(LineString(cl_pts_sc).coords)
                # 车道第一次转系：原始地图 --> Agent19
                ctrln_sc[:, 0:2] = (ctrln_sc[:, 0:2] - orig).dot(rot)
                # 追加列表---车道中线坐标点的全局坐标系（agent19）表征
                sc_vecs.append(ctrln_sc)

                cl_pts = []
                # 在一段细分的车道段内---均匀生成11个点
                for s in np.linspace(s_lb, s_ub, num_sub_segs + 1):
                    cl_pts.append(cl_ls.interpolate(s))
                ctrln = np.array(LineString(cl_pts).coords)
                # 车道第一次转系，先转移到以agent19(全局坐标系)为中心
                ctrln[:, 0:2] = (ctrln[:, 0:2] - orig).dot(rot)

                # 车道节点锚姿态表征
                # 沿轴进行均值计算---11个点位置求x，y均值，得到该车道段中心锚点
                anch_pos = np.mean(ctrln, axis=0)
                # 得到细分车道段的向量化表示：初始点指向最后一点，再用向量差除以其长度，以得到---锚点单位向量(大小为1，只有方向)表征
                anch_vec = (ctrln[-1] - ctrln[0]) / np.linalg.norm(ctrln[-1] - ctrln[0])
                # 锚点旋转矩阵
                anch_rot = np.array([[anch_vec[0], -anch_vec[1]],
                                     [anch_vec[1], anch_vec[0]]])

                # 车道锚姿态中心坐标点，车道锚姿态向量追加列表
                lane_ctrs.append(anch_pos)
                lane_vecs.append(anch_vec)

                # 车道第二次转系，所有车道坐标以自己中心和旋转矩阵进行旋转，转到各车道中心坐标系
                ctrln[:, 0:2] = (ctrln[:, 0:2] - anch_pos).dot(anch_rot)

                # 转系之后，ctrln为各车道锚坐标系下的中线坐标
                # 车道中线有11个插值点，形成10个向量，每个向量的中心点表征
                ctrs = np.asarray((ctrln[:-1] + ctrln[1:]) / 2.0, np.float32)
                # 各细分车道中线形成向量的10个中点拼接列表
                node_ctrs.append(ctrs)  # middle point

                # 车道中心线连接成多段向量表征，车道向量拼接列表
                vecs = np.asarray(ctrln[1:] - ctrln[:-1], np.float32)
                node_vecs.append(vecs)

                # 车道邻居语义信息表征：若为空，则为0，反之为1
                # ~ has left/right neighbor
                if lane.l_neighbor_id is None:
                    # w/o left neighbor
                    left.append(np.zeros(num_sub_segs, np.float32))
                else:
                    left.append(np.ones(num_sub_segs, np.float32))

                if lane.r_neighbor_id is None:
                    # w/o right neighbor
                    right.append(np.zeros(num_sub_segs, np.float32))
                else:
                    right.append(np.ones(num_sub_segs, np.float32))

                # 车道转向语义信息
                # ~ turn dir
                x = np.zeros((num_sub_segs, 2), np.float32)
                if lane.turn_direction == 'LEFT':
                    x[:, 0] = 1
                elif lane.turn_direction == 'RIGHT':
                    x[:, 1] = 1
                else:
                    pass
                turn.append(x)

                # 交通管制语义信息
                # ~ control & intersection
                control.append(lane.has_traffic_control * np.ones(num_sub_segs, np.float32))
                intersect.append(lane.is_intersection * np.ones(num_sub_segs, np.float32))


        # 各细化车道节点索引列表
        node_idcs = []  # List of range
        count = 0
        # node_ctrs细化车道中点列表
        # [0,1,2,3....]
        for i, ctr in enumerate(node_ctrs):
            node_idcs.append(range(count, count + len(ctr)))
            count += len(ctr)

        # 细化车道总节点数
        num_nodes = count
        # 细化车道总数
        num_lanes = len(node_idcs)

        lane_idcs = []  # node belongs to which lane, e.g. [0   0   0 ... 122 122 122]
        # 一条车道由细化后的各车道段组成，属于同一条车道id的细化车道，其id也相同
        for i, idcs in enumerate(node_idcs):
            lane_idcs.append(i * np.ones(len(idcs), np.int16))

        # lane contains which node
        # 该车道其中包含的细化车道节点
        for i in range(len(node_idcs)):
            node_idcs[i] = np.array(node_idcs[i])

        # 字典初始化
        graph = dict()
        # geometry
        # 细化车道各向量中心表征---（N，10，2）
        graph['node_ctrs'] = np.stack(node_ctrs, 0).astype(np.float32)
        # 细化车道各向量表征---（N，10，2）
        graph['node_vecs'] = np.stack(node_vecs, 0).astype(np.float32)
        # 全局坐标系（agent19）下的细化车道10个坐标点表征---（N，10，2）
        graph['sc_vecs'] = np.stack(sc_vecs, 0).astype(np.float32)

        # node features
        # 语义信息
        # 转向---（N，10，2）
        graph['turn'] = np.stack(turn, 0).astype(np.int16)
        # 管制，交叉路口，左右邻---（N，10）
        graph['control'] = np.stack(control, 0).astype(np.int16)
        graph['intersect'] = np.stack(intersect, 0).astype(np.int16)
        graph['left'] = np.stack(left, 0).astype(np.int16)
        graph['right'] = np.stack(right, 0).astype(np.int16)

        # 锚姿态表示：细分车道总中心点，细分车道总（首->尾）单位向量---（N，2）
        graph['lane_ctrs'] = np.array(lane_ctrs).astype(np.float32)
        graph['lane_vecs'] = np.array(lane_vecs).astype(np.float32)

        # node - lane
        # 总细分车道数（N）
        graph['num_lanes'] = num_lanes
        # 细分车道形成车道节点数（N*10）
        graph['num_nodes'] = num_nodes

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

        return graph

    # plotters
    def plot_trajs(self, ax, trajs, trajs_ctrs, trajs_vecs, pad_flags, orig, rot, vis_map=True):
        if not vis_map:
            rot = np.eye(2)
            orig = np.zeros(2)

        for i, (traj, ctr, vec) in enumerate(zip(trajs, trajs_ctrs, trajs_vecs)):
            zorder = 10
            if i == 0:
                clr = 'r'
                zorder = 20
            elif i == 1:
                clr = 'green'
            else:
                clr = 'orange'

            theta = np.arctan2(vec[1], vec[0])
            act_rot = np.array([[np.cos(theta), -np.sin(theta)],
                                [np.sin(theta), np.cos(theta)]])
            traj = traj.dot(act_rot.T) + ctr

            traj = traj.dot(rot.T) + orig
            ax.plot(traj[:, 0], traj[:, 1], marker='.', alpha=0.5, color=clr, zorder=zorder)
            ax.text(traj[self.args.obs_len, 0], traj[self.args.obs_len, 1], '{}'.format(i))
            ax.scatter(traj[:, 0], traj[:, 1], s=list((1 - pad_flags[i]) * 50 + 1), color='b')

    def plot_lane_graph(self, ax, city_name, orig, rot, lane_ids, lane_graph, vis_map=True):
        if vis_map:
            self.map_vis.show_map_with_lanes(ax, city_name, orig, lane_ids)
        else:
            rot = np.eye(2)
            orig = np.zeros(2)

        node_ctrs = lane_graph['node_ctrs']
        node_vecs = lane_graph['node_vecs']
        node_left = lane_graph['left']
        node_right = lane_graph['right']

        lane_ctrs = lane_graph['lane_ctrs']
        lane_vecs = lane_graph['lane_vecs']

        rel_lane_flags = lane_graph['rel_lane_flags']
        ax.plot(lane_ctrs[rel_lane_flags][:, 0], lane_ctrs[rel_lane_flags][:, 1], 'x', color='red', markersize=10)

        for ctrs_tmp, vecs_tmp, left_tmp, right_tmp, anch_pos, anch_vec in zip(node_ctrs, node_vecs,
                                                                               node_left, node_right,
                                                                               lane_ctrs, lane_vecs):
            anch_rot = np.array([[anch_vec[0], -anch_vec[1]],
                                 [anch_vec[1], anch_vec[0]]])
            ctrs_tmp = ctrs_tmp.dot(anch_rot.T) + anch_pos
            ctrs_tmp = ctrs_tmp.dot(rot.T) + orig

            vecs_tmp = vecs_tmp.dot(anch_rot.T)
            vecs_tmp = vecs_tmp.dot(rot.T)

            for j in range(vecs_tmp.shape[0]):
                vec = vecs_tmp[j]
                pt0 = ctrs_tmp[j] - vec / 2
                pt1 = ctrs_tmp[j] + vec / 2
                ax.arrow(pt0[0],
                         pt0[1],
                         (pt1-pt0)[0],
                         (pt1-pt0)[1],
                         edgecolor=None,
                         color='grey',
                         alpha=0.3,
                         width=0.1)

            anch_pos = anch_pos.dot(rot.T) + orig
            anch_vec = anch_vec.dot(rot.T)
            ax.plot(anch_pos[0], anch_pos[1], marker='*', color='cyan')
            ax.arrow(anch_pos[0], anch_pos[1], anch_vec[0], anch_vec[1], alpha=0.5, color='r', width=0.1)

            for i in range(len(left_tmp)):
                if left_tmp[i]:
                    ctr = ctrs_tmp[i]
                    vec = vecs_tmp[i] / np.linalg.norm(vecs_tmp[i])
                    vec = np.array([-vec[1], vec[0]])
                    ax.arrow(ctr[0],
                             ctr[1],
                             vec[0],
                             vec[1],
                             edgecolor=None,
                             color='red',
                             alpha=0.3,
                             width=0.05)

            for i in range(len(right_tmp)):
                if right_tmp[i]:
                    ctr = ctrs_tmp[i]
                    vec = vecs_tmp[i] / np.linalg.norm(vecs_tmp[i])
                    vec = np.array([vec[1], -vec[0]])
                    ax.arrow(ctr[0],
                             ctr[1],
                             vec[0],
                             vec[1],
                             edgecolor=None,
                             color='green',
                             alpha=0.3,
                             width=0.05)
