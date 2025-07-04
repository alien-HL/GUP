"""This script is for dataset preprocessing."""

import os
from os.path import expanduser
import time
from typing import Any, Dict, List, Tuple
import random
import argparse
from tqdm import tqdm
from joblib import Parallel, delayed
import multiprocessing
import numpy as np
import pandas as pd
import pickle as pkl
from argo_preprocess import ArgoPreproc

_FEATURES_SMALL_SIZE = 1024


def parse_arguments() -> Any:
    # 解析命令行参数
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        default="",
        type=str,
        help="Directory where the sequences (csv files) are saved",
    )
    parser.add_argument(
        "--save_dir",
        default="./dataset_argo/features/",
        type=str,
        help="Directory where the computed features are to be saved",
    )
    parser.add_argument(
        "--mode",
        required=True,
        type=str,
        help="train/val/test",
    )
    parser.add_argument(
        "--obs_len",
        default=20,
        type=int,
        help="Observed length of the trajectory",
    )
    parser.add_argument(
        "--pred_len",
        default=30,
        type=int,
        help="Prediction Horizon",
    )
    # 是否使用小数据集
    parser.add_argument(
        "--small",
        action="store_true",
        help="If true, a small subset of data is used.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="If true, debug mode.",
    )
    parser.add_argument(
        "--viz",
        action="store_true",
        help="If true, viz.",
    )
    return parser.parse_args()


# 加载文件，特征计算后存储
def load_seq_save_features(args: Any,
                           start_idx: int,
                           batch_size: int,
                           sequences: List[str],
                           save_dir: str, thread_idx: int) -> None:
    """ Load sequences, compute features, and save them """
    # 数据预处理函数实例化
    dataset = ArgoPreproc(args, verbose=False)

    # Enumerate over the batch starting at start_idx---从start_idx开始对批次进行枚举
    for seq in sequences[start_idx:start_idx + batch_size]:
        if not seq.endswith(".csv"):
            continue

        # 以.分割文件名，取第一个分割后的元素---作为每个文件场景id
        seq_id = int(seq.split(".")[0])
        # 数据集路径，拼接每一个文件名
        seq_path = f"{args.data_dir}/{seq}"
        # pandas读取此csv文件---（文件路径，表头），转化为数据框类型(DataFrame)
        df = pd.read_csv(seq_path, dtype={'TIMESTAMP': str,
                                          'TRACK_ID': str,
                                          'OBJECT_TYPE': str,
                                          'X': float,
                                          'Y': float,
                                          'CITY_NAME': str})

        # 数据执行预处理---ArgoPreproc.process()，返回两个字典型数据，data与headers
        data, headers = dataset.process(seq_id, df)

        if not args.debug:
            data_df = pd.DataFrame(data, columns=headers)
            filename = '{}'.format(data[0][0])
            data_df.to_pickle(f"{save_dir}/{filename}.pkl")  # compression='gzip'

    # 预处理完毕，输出
    print('Finish computing {} - {}'.format(start_idx, start_idx + batch_size))


if __name__ == "__main__":
    """Load sequences and save the computed features."""
    start = time.time()
    args = parse_arguments()

    # 获取数据集目录下的所有文件和子目录名
    sequences = os.listdir(args.data_dir)
    # 是否采用小数据集
    num_sequences = _FEATURES_SMALL_SIZE if args.small else len(sequences)
    sequences = sequences[:num_sequences]
    print("Num of sequences: ", num_sequences)

    # 多进程
    n_proc = multiprocessing.cpu_count() - 2 if not args.debug else 1

    # 批次数
    batch_size = np.max([int(np.ceil(num_sequences / n_proc)), 1])
    print('n_proc: {}, batch_size: {}'.format(n_proc, batch_size))

    # 预处理后数据存储路径
    save_dir = args.save_dir + f"{args.mode}"
    # 创建多层目录
    os.makedirs(save_dir, exist_ok=True)
    print('save processed dataset to {}'.format(save_dir))

    Parallel(n_jobs=n_proc)(delayed(load_seq_save_features)(args, i, batch_size, sequences, save_dir, k)
                            for i, k in zip(range(0, num_sequences, batch_size), range(len(range(0, num_sequences, batch_size)))))

    print(f"Preprocess for {args.mode} set completed in {(time.time()-start)/60.0:.2f} mins")
