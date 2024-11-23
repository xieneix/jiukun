import os
from scipy.io import loadmat
import numpy as np


def check_distance_file(file_path, distance_type='eucos'):
    """
    检查指定的距离文件是否存在、格式是否正确，以及内容是否符合预期。

    参数:
        file_path (str): 距离文件路径，例如 "../openmax/data/mean_distance_files/label00_distances.mat"
        distance_type (str): 距离类型，例如 'eucos'，默认值为 'eucos'

    返回:
        None
    """
    if not os.path.exists(file_path):
        print(f"文件不存在: {file_path}")
        return

    print(f"正在检查文件: {file_path}")

    try:
        # 加载文件内容
        data = loadmat(file_path)
        print(f"文件加载成功，包含的键: {list(data.keys())}")

        # 检查是否包含指定的距离类型
        if distance_type not in data:
            print(f"警告: 文件中不包含键 '{distance_type}'")
        else:
            # 提取指定类型的距离数据
            distance_scores = data[distance_type]
            print(f"'{distance_type}' 数据的形状: {distance_scores.shape}")
            print(
                f"数据统计信息 - 最小值: {distance_scores.min()}, 最大值: {distance_scores.max()}, 均值: {distance_scores.mean()}")

            # 检查维度是否符合预期
            if distance_scores.ndim != 2:
                print(f"警告: 数据维度异常，期望 2 维，但实际为 {distance_scores.ndim} 维")
            else:
                print(f"数据维度正确，形状为 {distance_scores.shape}")

    except Exception as e:
        print(f"加载文件时发生错误: {e}")


def check_mav(file_path):
    print(f"正在检查 MAV 文件: {file_path}")
    try:
        data = loadmat(file_path)
        print(f"文件加载成功，包含的键: {list(data.keys())}")

        # 尝试访问 'mean_vec' 键
        if 'mean_vec' in data:
            mav = data['mean_vec']
            print(f"'mean_vec' 数据的形状: {mav.shape}")
            print(f"数据统计信息 - 最小值: {np.min(mav)}, 最大值: {np.max(mav)}, 均值: {np.mean(mav)}")
        else:
            print("文件中不包含 'mean_vec' 键，请确认 MAV 数据是否存在。")

    except Exception as e:
        print(f"无法加载 MAV 文件: {e}")

# 调用函数检查指定的文件
# file_path = "../openmax/data/mean_distance_files/label00_distances.mat"
# check_distance_file(file_path, distance_type='eucos')

mav_file_path = "../openmax/data/mean_files/label01.mat"
check_mav(mav_file_path)

