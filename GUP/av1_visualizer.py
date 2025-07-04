import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

from utils.vis_utils import ArgoMapVisualizer
from matplotlib.patches import FancyArrowPatch


class Visualizer():
    def __init__(self):
        self.map_vis = ArgoMapVisualizer()

    def draw_once(self, post_out, data, eval_out, show_map=False, test_mode=False, split='val'):
        batch_size = len(data['SEQ_ID'])

        seq_id = data['SEQ_ID'][0]
        city_name = data['CITY_NAME'][0]
        orig = data['ORIG'][0]
        rot = data['ROT'][0]
        trajs_obs = data['TRAJS_OBS'][0]
        trajs_fut = data['TRAJS_FUT'][0]
        pads_obs = data['PAD_OBS'][0]
        pads_fut = data['PAD_FUT'][0]
        trajs_ctrs = data['TRAJS_CTRS'][0]
        trajs_vecs = data['TRAJS_VECS'][0]
        lane_graph = data['LANE_GRAPH'][0]

        res_cls = post_out['out_sc'][0]
        res_reg = post_out['out_sc'][1]

        _, ax = plt.subplots(figsize=(12, 12))
        ax.axis('equal')
        ax.set_title('{}-{}'.format(seq_id, city_name))

        if show_map:
            self.map_vis.show_surrounding_elements(ax, city_name, orig)
        else:
            rot = torch.eye(2)
            orig = torch.zeros(2)

        # 历史轨迹绘制
        # trajs
        for i, (traj_obs, pad_obs, ctr, vec) in enumerate(zip(trajs_obs, pads_obs, trajs_ctrs, trajs_vecs)):
            zorder = 30
            # 最感兴趣Agent---红色
            if i == 0:
                clr = 'tomato'
                zorder = 30
            # 自动驾驶车辆AV---蓝色
            elif i == 1:
                clr = 'tomato'
            else:
                # 其他车辆
                clr = 'tomato'

            # if torch.sum(pad_obs) < 15:
            if torch.sum(pads_obs[i]) < 15 or torch.sum(pads_fut[i]) < 30:
                # 步长很短车辆
                clr = 'grey'

            theta = np.arctan2(vec[1], vec[0])
            act_rot = torch.Tensor([[np.cos(theta), -np.sin(theta)],
                                    [np.sin(theta), np.cos(theta)]])

            traj_obs = torch.matmul(traj_obs, act_rot.T) + ctr
            traj_obs = torch.matmul(traj_obs, rot.T) + orig
            # 堆叠方式 --- ./。    markersize 设置标记（点）的大小      zorder堆叠顺序，越大会覆盖下边的
            ax.plot(traj_obs[:, 0], traj_obs[:, 1], marker='.', alpha=0.4, color=clr, zorder=zorder)
            # 最后一步
            ax.plot(traj_obs[-1, 0], traj_obs[-1, 1], marker='o', alpha=0.4, color=clr, zorder=zorder, markersize=10)

        if not test_mode:
            # if not test mode, vis GT trajectories
            # 地面标签绘制
            for i, (traj_fut, pad_fut, ctr, vec) in enumerate(zip(trajs_fut, pads_fut, trajs_ctrs, trajs_vecs)):
                zorder = 20
                # 最感兴趣agent真实地面标签---粉色
                if i == 0:
                    clr = 'cyan'
                    # 使其置于最上面
                    zorder = 20
                # 其他车辆（AV+others）真实标签全部为绿色
                elif i == 1:
                    clr = 'cyan'
                #else:
                #    clr = 'cyan'

                if torch.sum(pads_obs[i]) < 15 or torch.sum(pads_fut[i]) < 30:
                    continue

                theta = np.arctan2(vec[1], vec[0])
                act_rot = torch.Tensor([[np.cos(theta), -np.sin(theta)],
                                        [np.sin(theta), np.cos(theta)]])

                traj_fut = torch.matmul(traj_fut, act_rot.T) + ctr
                traj_fut = torch.matmul(traj_fut, rot.T) + orig
                ax.plot(traj_fut[:, 0], traj_fut[:, 1], alpha=0.5, color=clr, zorder=zorder)

                mk = '*' if torch.sum(pad_fut) == 30 else '*'
                # 最后一步
                ax.plot(traj_fut[-1, 0], traj_fut[-1, 1], marker=mk, alpha=0.8, color=clr, zorder=50, markersize=12)

        # 车辆未来轨迹绘制
        # traj pred all
        # print('res_reg: ', [x.shape for x in res_reg])
        res_reg = res_reg[0].cpu().detach()
        res_cls = res_cls[0].cpu().detach()
        for i, (trajs, probs, ctr, vec) in enumerate(zip(res_reg, res_cls, trajs_ctrs, trajs_vecs)):
            zorder = 10
            # 最感兴趣agent---红色
            if i == 0:
                clr = 'blanchedalmond'
                zorder = 10
            # 自动驾驶车辆AV---蓝色
            elif i == 1:
                clr = 'blanchedalmond'
            # others---深蓝色
            #else:
            #    clr = 'blanchedalmond'

            if torch.sum(pads_obs[i]) < 15 or torch.sum(pads_fut[i]) < 30:
                continue

            theta = np.arctan2(vec[1], vec[0])
            act_rot = torch.Tensor([[np.cos(theta), -np.sin(theta)],
                                    [np.sin(theta), np.cos(theta)]])

            for traj, prob in zip(trajs, probs):
                if prob < 0.05 and (not i in [0, 1]):
                    continue
                traj = torch.matmul(traj, act_rot.T) + ctr
                # 画多少轨迹
                traj = torch.matmul(traj, rot.T) + orig
                ax.plot(traj[:-1, 0], traj[:-1, 1], alpha=0.7, color=clr, zorder=zorder, linestyle='--')
                # ax.plot(traj[-1, 0], traj[-1, 1], alpha=0.3, marker='o', color=clr, zorder=zorder, markersize=12)

                # 箭头绘制
                ax.arrow(traj[-2, 0],
                         traj[-2, 1],
                         (traj[-2, 0] - traj[-3, 0]),
                         (traj[-2, 1] - traj[-3, 1]),
                         edgecolor=None,
                         color=clr,
                         alpha=0.4,
                         width=0.3,
                         zorder=zorder)


                prob_value = prob.item()
                prob = f"{prob_value:.2f}"  # 格式化为百分数，保留两位小数

                # 在箭头附近添加注释，显示百分数
                ax.annotate(prob,  # 注释文本
                            xy=(traj[-1, 0], traj[-1, 1]),  # 注释的坐标（箭头的末端）
                            xytext=(traj[-1, 0] + 0.1, traj[-1, 1] + 0.1),  # 注释文本的位置，稍微偏移
                            fontsize=5,  # 字体大小
                            color=clr,  # 文字颜色
                            ha='left',  # 水平对齐方式
                            va='bottom',  # 垂直对齐方式
                            zorder=zorder+1)  # 注释层级高于箭头


        # lane graph
        node_ctrs = lane_graph['node_ctrs']  # [196, 10, 2]
        node_vecs = lane_graph['node_vecs']  # [196, 10, 2]
        lane_ctrs = lane_graph['lane_ctrs']  # [196, 2]
        lane_vecs = lane_graph['lane_vecs']  # [196, 2]

        for ctrs_tmp, vecs_tmp, anch_pos, anch_vec in zip(node_ctrs, node_vecs, lane_ctrs, lane_vecs):
            anch_rot = torch.Tensor([[anch_vec[0], -anch_vec[1]],
                                     [anch_vec[1], anch_vec[0]]])
            ctrs_tmp = torch.matmul(ctrs_tmp, anch_rot.T) + anch_pos
            ctrs_tmp = torch.matmul(ctrs_tmp, rot.T) + orig
            # 车道中线画图
            # （横坐标，纵坐标，alpha透明度，linestyle='dotted': 设置线条的样式为点线（dotted），color='grey': 设置线条的颜色为灰色。）
            ax.plot(ctrs_tmp[:, 0], ctrs_tmp[:, 1], alpha=0.2, linestyle='dotted', color='white')

        plt.tight_layout()
        plt.show()
