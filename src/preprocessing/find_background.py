import numpy as np
from numba import jit, prange
from tqdm import tqdm

from ..utils.io import read_bin
from ..utils.visualization import visualize_dynamic_points


@jit(nopython=True, parallel=True)
def calculate_reliability(ranges, reflectivity):
    """
    使用Numba加速的LiDAR点可靠度模型，考虑0-100%范围的反射率
    参数:
    - ranges: 距离值（米）
    - reflectivity: 反射率值（0-255）
    返回:
    - 位置可靠度值（0-1范围内）
    """
    # 初始化可靠度数组
    reliability = np.zeros_like(ranges, dtype=np.float32)
    min_std = 0.5

    # 预计算常量 - 扩展到完整范围
    a_low = (3-min_std) / (90.0 * 90.0)  # 10%反射率时的系数
    a_high = (0.8-min_std) / (90.0 * 90.0)  # 90%反射率时的系数
    refl_factor = (a_low - a_high) / (90.0 - 10.0)  # 预计算插值因子

    # 计算0%和100%的系数
    a_at_0_percent = a_low + 10.0 * refl_factor
    a_at_100_percent = max(0.0, a_high - 10.0 * refl_factor)  # 确保不为负

    for i in prange(len(ranges)):  # 使用prange进行并行计算
        r = ranges[i]

        # 将反射率归一化到百分比
        refl = (reflectivity[i] / 255.0) * 100.0

        # 根据反射率计算系数（不需要限制范围）
        if refl == 0.0:
            a = a_at_0_percent
        elif refl < 10.0:
            # 0-10%范围内线性插值
            t = refl / 10.0
            a = a_at_0_percent * (1.0 - t) + a_low * t
        elif refl == 10.0:
            # 10%反射率时的系数
            a = a_low
        elif 10 < refl < 90.0:
            # 10-90%范围使用原来的线性插值
            a = a_low - (refl - 10.0) * refl_factor
        elif refl == 90.0:
            # 90%反射率时的系数
            a = a_high
        elif refl < 100.0:
            # 90-100%范围内线性插值
            t = (refl - 90.0) / 10.0
            a = a_high * (1.0 - t) + a_at_100_percent * t
        else:
            a = a_at_100_percent

        # 二次函数计算标准差（厘米）
        precision_std = a * (r * r) + min_std

        # 计算可靠度
        reliability[i] = max(0.0, 1.0 - 0.01 * precision_std)

    return reliability


@jit(nopython=True, parallel=True)
def update_background_model(mean, std, weight, mean_reflectivity, counts,
                            indices, ranges, reliability, reflectivity, learning_rate, total_points):
    for i in prange(len(indices)):
        idx = indices[i]
        if idx >= total_points:
            continue
        r = ranges[i]
        rel = reliability[i]
        refl = reflectivity[i]
        # 更新计数
        counts[idx] += 1
        # 首次观测
        if weight[idx] == 0:
            mean[idx] = r
            weight[idx] = rel
            mean_reflectivity[idx] = refl
        else:
            # 计算自适应学习率
            adaptive_lr = learning_rate * rel
            # 计算差异
            diff = r - mean[idx]
            diff_refl = refl - mean_reflectivity[idx]
            # 更新参数
            weight[idx] += adaptive_lr
            mean[idx] += adaptive_lr * diff / weight[idx]
            mean_reflectivity[idx] += adaptive_lr * diff_refl / weight[idx]
            std[idx] = np.sqrt((1 - adaptive_lr) * (std[idx] ** 2) + adaptive_lr * (diff ** 2))
    return mean, std, weight, mean_reflectivity, counts


@jit(nopython=True, parallel=True)
def detect_foreground_points(indices, ranges, reliability, reflectivity, mean, std, mean_reflectivity,
                             counts, min_count, base_threshold, threshold_refl, foreground_mask, range_factor):
    """
    使用Numba加速的前景点检测
    """
    for i in prange(len(indices)):
        idx = indices[i]
        if idx >= len(mean):
            continue
        r = ranges[i]
        rel = reliability[i]
        refl = reflectivity[i]
        if counts[idx] >= min_count and mean[idx] > 0:
            # 计算差异
            diff = abs(mean[idx] - r)* rel
            # 计算自适应阈值
            adaptive_threshold = base_threshold + std[idx] * (1 - rel) * range_factor
            adaptive_threshold = base_threshold + std[idx]
            diff_refl = abs(refl - mean_reflectivity[idx])
            # 检测前景点
            if diff > adaptive_threshold and diff_refl > threshold_refl:
                foreground_mask[indices[i]] = True
    return foreground_mask


def build_background(pcd_files, min_range=0.5, max_range=200,
                     learning_rate=0.01, init_std=1, sample_size=1000,
                     columns=['x', 'y', 'z', 'range', 'reflectivity', 'near_ir', 'intensity', 'flags']):
    # 初始化模型参数
    total_points = len(read_bin(pcd_files[0], len(columns)))
    mean = np.zeros(total_points, dtype=np.float32)
    std = np.ones(total_points, dtype=np.float32) * init_std
    weight = np.zeros(total_points, dtype=np.float32)
    mean_reflectivity = np.zeros(total_points, dtype=np.float32)
    counts = np.zeros(total_points, dtype=np.int32)
    # 计算采样间隔
    interval = max(1, len(pcd_files) // sample_size)
    # 处理每个采样帧
    for i in tqdm(range(0, len(pcd_files), interval), desc="Building background model"):
        # 读取点云数据
        pcd_array = read_bin(pcd_files[i], len(columns))
        # 提取数据
        ranges = pcd_array[:, columns.index('range')]
        reflectivity = pcd_array[:, columns.index('reflectivity')]
        # 创建点索引
        indices = np.arange(len(pcd_array))
        # 过滤有效点
        mask = (ranges > min_range) & (ranges < max_range)
        valid_ranges = ranges[mask]
        valid_reflectivity = reflectivity[mask]
        valid_indices = indices[mask]

        # 使用新的可靠度模型
        valid_reliability = calculate_reliability(valid_ranges, valid_reflectivity)

        # Numba加速的模型更新
        mean, std, weight, mean_reflectivity, counts = update_background_model(
            mean, std, weight, mean_reflectivity, counts,
            valid_indices, valid_ranges, valid_reliability, valid_reflectivity, learning_rate, total_points
        )
    # 打包背景模型
    bg_model = np.column_stack([mean, std, mean_reflectivity, counts])
    # 输出模型统计信息
    valid_points = counts > 0
    print(f"背景模型构建完成: {np.sum(valid_points)}/{total_points} 有效点 ({np.sum(valid_points) / total_points:.2%})")
    return bg_model

def detect_foreground(pcd_array, bg_model, base_threshold=1, base_threshold_refl=0, min_count=10, min_range=0.5,
                      max_range=200, range_factor=1,
                      visualize=False, bg_model_columns=['mean', 'std', 'mean_reflectivity', 'counts'],
                      columns=['x', 'y', 'z', 'range', 'reflectivity', 'near_ir', 'intensity', 'flags']):
    # 从背景模型提取参数
    mean = bg_model[:, bg_model_columns.index('mean')]
    std = bg_model[:, bg_model_columns.index('std')]
    mean_reflectivity = bg_model[:, bg_model_columns.index('mean_reflectivity')]
    counts = bg_model[:, bg_model_columns.index('counts')]
    # 从点云提取数据
    ranges = pcd_array[:, columns.index('range')]
    reflectivity = pcd_array[:, columns.index('reflectivity')]
    # 创建点索引和掩码
    indices = np.arange(len(pcd_array))
    mask = (ranges > min_range) & (ranges < max_range)
    valid_ranges = ranges[mask]
    valid_reflectivity = reflectivity[mask]
    valid_indices = indices[mask]

    # 使用新的可靠度模型
    valid_reliability = calculate_reliability(valid_ranges, valid_reflectivity)

    # 可靠度统计
    # min_reliability = np.min(valid_reliability)
    # max_reliability = np.max(valid_reliability)
    # mean_reliability = np.mean(valid_reliability)
    # median_reliability = np.median(valid_reliability)
    # std_reliability = np.std(valid_reliability)
    # print(f"可靠度统计: min={min_reliability:.6f}, max={max_reliability:.6f}, mean={mean_reliability:.6f}, median={median_reliability:.6f}, std={std_reliability:.6f}")
    # min_reliability_index = np.argmin(valid_reliability)
    # max_reliability_index = np.argmax(valid_reliability)
    # min_reliability_range = valid_ranges[min_reliability_index]
    # max_reliability_range = valid_ranges[max_reliability_index]
    # print(f"最小可靠度点: range={min_reliability_range:.6f}, reflectivity={valid_reflectivity[min_reliability_index]:.6f}")
    # print(f"最大可靠度点: range={max_reliability_range:.6f}, reflectivity={valid_reflectivity[max_reliability_index]:.6f}")

    # 初始化所有点的可靠度为0
    all_reliability = np.zeros(len(pcd_array), dtype=np.float32)
    all_reliability[mask] = valid_reliability

    # 初始化前景掩码
    foreground_mask = np.zeros(len(pcd_array), dtype=np.bool_)
    # Numba加速的前景检测
    foreground_mask = detect_foreground_points(
        valid_indices, valid_ranges, valid_reliability, valid_reflectivity,
        mean, std, mean_reflectivity, counts, min_count, base_threshold, base_threshold_refl, foreground_mask,
        range_factor
    )
    # 可视化结果
    if visualize:
        dynamic_points = pcd_array[foreground_mask]
        static_points = pcd_array[~foreground_mask]
        visualize_dynamic_points(dynamic_points, static_points)
    return foreground_mask, all_reliability