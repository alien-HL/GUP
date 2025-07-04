import os
import sys
import time
from typing import Any, Dict, List, Tuple, Union
from datetime import datetime
import argparse
import numpy as np
import faulthandler
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from loader import Loader
from utils.utils import AverageMeter, AverageMeterForDict
from argoverse.evaluation.competition_util import generate_forecasting_h5





def parse_arguments() -> Any:
    """Arguments for running the baseline.

    Returns:
        parsed arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="test", type=str, help="Mode, train/val/test")
    parser.add_argument("--features_dir", required=True, default="", type=str, help="Path to the dataset")
    parser.add_argument("--train_batch_size", type=int, default=16, help="Training batch size")
    parser.add_argument("--test_batch_size", type=int, default=32, help="Test batch size")
    parser.add_argument("--use_cuda", action="store_true", help="Use CUDA for acceleration")
    parser.add_argument("--data_aug", action="store_true", help="Enable data augmentation")
    parser.add_argument("--adv_cfg_path", required=True, default="", type=str)
    parser.add_argument("--model_path", required=False, type=str, help="path to the saved model")
    return parser.parse_args()

def gpu(data):
    """
    Transfer tensor in `data` to gpu recursively
    `data` can be dict, list or tuple
    """
    if isinstance(data, list) or isinstance(data, tuple):
        data = [gpu(x) for x in data]
    elif isinstance(data, dict):
        data = {key:gpu(_data) for key,_data in data.items()}
    elif isinstance(data, torch.Tensor):
        data = data.contiguous().cuda(non_blocking=True)
    return data


def main():
    args = parse_arguments()
    print('Args: {}\n'.format(args))

    faulthandler.enable()

    if args.use_cuda and torch.cuda.is_available():
        device = torch.device("cuda", 0)
    else:
        device = torch.device('cpu')

    if not args.model_path.endswith(".tar"):
        assert False, "Model path error - '{}'".format(args.model_path)

    loader = Loader(args, device, is_ddp=False)
    print('[Resume] Loading state_dict from {}'.format(args.model_path))
    loader.set_resume(args.model_path)
    test_set, net, loss_fn, _, evaluator = loader.load()

    dl_test = DataLoader(test_set,
                         batch_size=args.test_batch_size,
                         shuffle=True,
                         num_workers=8,
                         collate_fn=test_set.collate_fn,
                         drop_last=False,
                         pin_memory=True)

    net.eval()

    # 测试集推理
    traj = {}
    prob = {}
    cities = {}

    with torch.no_grad():
        # Test
        test_start = time.time()
        test_eval_meter = AverageMeterForDict()

        for i, data in enumerate(tqdm(dl_test)):
            # 数据预处理
            data_in = net.pre_process(data)
            # 模型拟合
            out = net(data_in)

            # 无需计算损失
            # _ = loss_fn(out, data)

            # 经过后处理，只拿到最感兴趣目标agent信息，下面作坐标系变换操作
            post_out = net.post_process(out)

            # 生成各场景文件感兴趣agent多模态轨迹：此刻已在Agent坐标系（全局坐标系），只需用逆矩阵相乘再平移原点即可转回原始地图坐标系
            # results = [x.detach().cpu().numpy() for x in post_out["traj_pred"]]
            results = post_out["traj_pred"]
            pi = post_out['prob_pred']

            # prediction
            # 数据处理时无原始 -> AV坐标系转换中间步。而是直接将原始->感兴趣Agent坐标系，作为全局坐标系，后再转到各车辆局部坐标系
            # 对于旋转矩阵，转换回去需要变换逆矩阵，二阶逆矩阵主对角线不变，副对角线取反
            rot, orig = gpu(data["ROT"]), gpu(data["ORIG"])
            # print(rot[0])

            # 求逆矩阵，追加列表
            rot_inverse = []
            for i in rot:
                # 注：2024.12.10---旋转矩阵的逆矩阵直接就是他的转置
                rot = torch.inverse(i)
                rot_inverse.append(rot)
            # print(rot_inverse)

            # transform prediction to world coordinates
            # 将agent19坐标系向地图坐标系转换
            for i in range(len(results)):
                # 得到原始地图坐标系的预测轨迹
                results[i] = torch.matmul(results[i], rot_inverse[i]) + orig[i].view(1, 1, 1, -1)
            # print(results)

            results = results.cpu().numpy()
            pi = pi.cpu().numpy()

            for i, (argo_idx, pred_traj, pred_prob) in enumerate(zip(data["SEQ_ID"], results, pi)):
                traj[argo_idx] = pred_traj.squeeze()
                cities[argo_idx] = data["CITY_NAME"][i]
                prob[argo_idx] = pred_prob

        # 榜单提交API---生成测试集文件h5
        print('开始生成h5文件')
        output_path = './test_output'
        filename = 'submission'
        generate_forecasting_h5(traj, output_path, filename, prob)
        print('\nh5文件已成功生成')



if __name__ == "__main__":
    main()