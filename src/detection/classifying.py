# import numpy as np
# from numba import jit, prange
#
# from .compute_bbox import compute_bbox
#
#
# def classifying_stage_1(clusters_coords):
#     """
#     执行简单分类，将物体分为"车辆部件"和"其他"两类
#
#     Args:
#         clusters_coords: 每个聚类的坐标列表
#
#     Returns:
#         coarse_classes: 类别标签列表 (0: 其他, 1: 车辆部件)
#     """
#     coarse_classes = []
#
#     for cluster_points in clusters_coords:
#         # 提取特征
#         try:
#             # 计算边界框
#             bbox = compute_bbox(cluster_points)
#             n_points = len(cluster_points)
#             x, y, z, width, length, height, _ = bbox
#             # 计算基本特征
#             volume = width * length * height
#             density = n_points / volume if volume > 0.0001 else 0
#             aspect_ratio = max(width, length) / (min(width, length) + 1e-6)  # 避免除零
#             # 计算高度分布
#             # 重心高度
#             min_height = np.min(cluster_points[:, 2]) if len(cluster_points) > 0 else 0
#             max_height = np.max(cluster_points[:, 2]) if len(cluster_points) > 0 else 0
#             height_mean = 0.5 * (min_height + max_height) if len(cluster_points) > 0 else 0
#             height_std = np.std(cluster_points[:, 2]) if len(cluster_points) > 1 else 0
#             # 计算协方差矩阵和主成分
#             if len(cluster_points) >= 3:
#                 cov_matrix = np.cov(cluster_points[:, :2].T)  # 只用XY平面
#                 if cov_matrix.shape == (2, 2):
#                     eigenvalues = np.sort(np.abs(np.linalg.eigvals(cov_matrix)))[::-1]
#                     xy_ratio = eigenvalues[0] / (eigenvalues[1] + 1e-6)  # XY平面主轴比
#                 else:
#                     xy_ratio = 1.0
#             else:
#                 xy_ratio = 1.0
#             # 车辆部件评分
#             vehicle_part_score = 0
#             # 密度判断
#             if density < 10:  # 密度太低可能是疏散的树叶等
#                 vehicle_part_score -= 10
#             elif density > 100:  # 密度很高的部分可能是车辆的特定部件
#                 vehicle_part_score += 1.0
#             # 1. 体积范围检查
#             if 8 <= volume <= 120.0:
#                 vehicle_part_score += 2.0
#             elif volume < 1.0:
#                 vehicle_part_score -= 5.0
#             # 2. 长宽比检查（车辆通常是细长的）
#             if 1.5 <= aspect_ratio <= 4:
#                 vehicle_part_score += 1.5
#             elif aspect_ratio > 4:
#                 vehicle_part_score -= 1.0
#             # 3. 高度检查（车辆高度通常在一定范围内）
#             if 0.5 <= height <= 3.0:
#                 vehicle_part_score += 1.0
#             else:
#                 vehicle_part_score -= 5
#             # 4. 高度分布（车辆部件通常高度变化较小）
#             if height_std < 0.3 * height:
#                 vehicle_part_score += 1.0
#             # 5. 主轴比（车辆在XY平面通常有明显的方向性）
#             if xy_ratio > 2.0:
#                 vehicle_part_score += 1.5
#             # 6. 重心高度（车辆通常贴近地面）
#             if 0.5<=height_mean < 1.8:
#                 vehicle_part_score += 1.0
#             else:
#                 vehicle_part_score -= 10
#             # 7. 尺寸组合（车辆典型尺寸组合）
#             if (1.5 <= width <= 2.5 and 3.5 <= length <= 6.0) or \
#                     (2.0 <= width <= 3.0 and 5.0 <= length <= 15.0):
#                 vehicle_part_score += 2.0
#             # 设置阈值
#             is_vehicle_part = vehicle_part_score >= 5.0
#             if is_vehicle_part:
#                 coarse_classes.append(1)  # 车辆部件
#             else:
#                 coarse_classes.append(0)  # 其他
#         except Exception as e:
#             print(f"Error in coarse classification: {e}")
#             coarse_classes.append(0)  # 出错时默认为"其他"
#     return coarse_classes
#
#
# def classifying_stage_2(clusters_coords, coarse_classes=None):
#     """
#     使用效用函数和softmax的分类方法，但采用更直接的特征判断
#
#     Args:
#         clusters_coords: 每个集群的坐标
#         coarse_classes: 可选的粗分类结果
#
#     Returns:
#         cluster_classes: 类别标签列表
#         class_probabilities: 类别概率分布列表
#     """
#     cluster_classes = []
#     class_probabilities = []
#
#     for i, cluster_points in enumerate(clusters_coords):
#         try:
#             # 提取特征
#             bbox = compute_bbox(cluster_points)
#             x, y, z, width, length, height, _ = bbox
#             volume = width * length * height
#             n_points = len(cluster_points)
#             density = n_points / volume if volume > 0.0001 else 0
#             length = max(length, width)
#             width = min(length, width)
#             # 计算比率
#             aspect_ratio = length /(width + 1e-6)
#             height_ratio = height / (length + 1e-6)
#
#             # 初始化效用分数
#             utility_scores = np.zeros(4)  # [人, 小型车, 大型车, 其他]
#
#             # ======== 人的效用分数 ========
#             if 1 <= height <= 2.0:
#                 utility_scores[0] += 2
#             if width <= 0.8 and length <= 0.8:
#                 utility_scores[0] += 3
#             if height_ratio >= 1.5:
#                 utility_scores[0] += 3
#             if volume < 1.0:
#                 utility_scores[0] += 2
#             if density >= 10:
#                 utility_scores[0] += 1
#             else:
#                 utility_scores[0] -= 2
#             if 0.5<=z < 1.5:
#                 utility_scores[0] += 1
#             else:
#                 utility_scores[0] -= 2
#
#             # ======== 小型车效用分数 ========
#             if 1.0 <= height <= 2.2:
#                 utility_scores[1] += 1
#             if 1.5 <= width <= 2.5:
#                 utility_scores[1] += 2
#             if 2.5 <= length <= 4:
#                 utility_scores[1] += 2
#             if 1.5 <= aspect_ratio <= 4.0:
#                 utility_scores[1] += 2
#             if 8<volume <= 40:
#                 utility_scores[1] += 3
#             if density >= 10:
#                 utility_scores[1] += 1
#             else:
#                 utility_scores[1] -= 2
#             if 0.5<=z < 1.5:
#                 utility_scores[1] += 1
#             else:
#                 utility_scores[1] -= 2
#
#             # ======== 大型车效用分数 ========
#             if 2.5 <= height <= 4.5:
#                 utility_scores[2] += 1
#             if 2.0 <= width <= 3.0:
#                 utility_scores[2] += 2
#             if 5.0 <= length <= 15.0:
#                 utility_scores[2] += 2
#             if aspect_ratio >= 2.0:
#                 utility_scores[2] += 2
#             if volume > 40.0:
#                 utility_scores[2] += 3
#             if density >= 10:
#                 utility_scores[2] += 1
#             else:
#                 utility_scores[2] -= 2
#             if 1<= z < 2.0:
#                 utility_scores[2] += 1
#             else:
#                 utility_scores[2] -= 2
#
#             # 应用阈值以筛选明确的类别
#             # 如果最高分的类别不是"其他"，且分数太低，则改为"其他"
#             max_class = np.argmax(utility_scores)
#             if utility_scores[max_class] < 6:
#                 utility_scores[3] += 12
#
#             # 应用softmax获取概率
#             temperature = 1.0  # 控制概率分布的锐度
#             utility_scores = utility_scores / temperature
#             # 为避免数值溢出，减去最大效用分数
#             utility_scores = utility_scores - np.max(utility_scores)
#             exp_scores = np.exp(utility_scores)
#             probs = exp_scores / np.sum(exp_scores)
#
#             # 获取预测类别（最高概率）
#             class_label = np.argmax(probs)
#
#             # 存储结果
#             cluster_classes.append(class_label)
#             class_probabilities.append(probs)
#
#         except Exception as e:
#             print(f"Error classifying cluster {i}: {e}")
#             # 默认分类为"其他"
#             cluster_classes.append(3)
#             class_probabilities.append(np.array([0.0, 0.0, 0.0, 1.0]))
#     if class_probabilities:
#         class_probabilities = np.max(class_probabilities, axis=1)
#     else:
#         class_probabilities = np.array([])
#     return cluster_classes, class_probabilities


import numpy as np
from numba import jit, prange

from .compute_bbox import compute_bbox


def classifying_stage_1(clusters_coords):
    """
    执行简单分类，将物体分为"车辆部件"和"其他"两类

    Args:
        clusters_coords: 每个聚类的坐标列表

    Returns:
        coarse_classes: 类别标签列表 (0: 其他, 1: 车辆部件)
    """
    coarse_classes = []

    for cluster_points in clusters_coords:
        # 提取特征
        try:
            # 计算边界框
            bbox = compute_bbox(cluster_points)
            n_points = len(cluster_points)
            x, y, z, width, length, height, _ = bbox
            # 计算基本特征
            volume = width * length * height
            density = n_points / volume if volume > 0.0001 else 0
            aspect_ratio = max(width, length) / (min(width, length) + 1e-6)  # 避免除零
            # 计算高度分布
            # 重心高度
            min_height = np.min(cluster_points[:, 2]) if len(cluster_points) > 0 else 0
            max_height = np.max(cluster_points[:, 2]) if len(cluster_points) > 0 else 0
            height_mean = 0.5 * (min_height + max_height) if len(cluster_points) > 0 else 0
            height_std = np.std(cluster_points[:, 2]) if len(cluster_points) > 1 else 0
            # 计算协方差矩阵和主成分
            if len(cluster_points) >= 3:
                cov_matrix = np.cov(cluster_points[:, :2].T)  # 只用XY平面
                if cov_matrix.shape == (2, 2):
                    eigenvalues = np.sort(np.abs(np.linalg.eigvals(cov_matrix)))[::-1]
                    xy_ratio = eigenvalues[0] / (eigenvalues[1] + 1e-6)  # XY平面主轴比
                else:
                    xy_ratio = 1.0
            else:
                xy_ratio = 1.0
            # 车辆部件评分
            vehicle_part_score = 0
            # 密度判断
            if density < 10:  # 密度太低可能是疏散的树叶等
                vehicle_part_score -= 10
            elif density > 100:  # 密度很高的部分可能是车辆的特定部件
                vehicle_part_score += 1.0
            # 1. 体积范围检查
            if 8 <= volume <= 120.0:
                vehicle_part_score += 2.0
            elif volume < 1.0:
                vehicle_part_score -= 5.0
            # 2. 长宽比检查（车辆通常是细长的）
            if 1.5 <= aspect_ratio <= 4:
                vehicle_part_score += 1.5
            elif aspect_ratio > 4:
                vehicle_part_score -= 1.0
            # 3. 高度检查（车辆高度通常在一定范围内）
            if 0.5 <= height <= 3.0:
                vehicle_part_score += 1.0
            else:
                vehicle_part_score -= 5
            # 4. 高度分布（车辆部件通常高度变化较小）
            if height_std < 0.3 * height:
                vehicle_part_score += 1.0
            # 5. 主轴比（车辆在XY平面通常有明显的方向性）
            if xy_ratio > 2.0:
                vehicle_part_score += 1.5
            # 6. 重心高度（车辆通常贴近地面）
            if 0.5<=height_mean < 1.8:
                vehicle_part_score += 1.0
            else:
                vehicle_part_score -= 10
            # 7. 尺寸组合（车辆典型尺寸组合）
            if (1.5 <= width <= 2.5 and 3.5 <= length <= 6.0) or \
                    (2.0 <= width <= 3.0 and 5.0 <= length <= 15.0):
                vehicle_part_score += 2.0
            # 设置阈值
            is_vehicle_part = vehicle_part_score >= 5.0
            if is_vehicle_part:
                coarse_classes.append(1)  # 车辆部件
            else:
                coarse_classes.append(0)  # 其他
        except Exception as e:
            print(f"Error in coarse classification: {e}")
            coarse_classes.append(0)  # 出错时默认为"其他"
    return coarse_classes


def classifying_stage_2(clusters_coords, coarse_classes=None):
    """
    使用效用函数和softmax的分类方法，但采用更直接的特征判断

    Args:
        clusters_coords: 每个集群的坐标
        coarse_classes: 可选的粗分类结果

    Returns:
        cluster_classes: 类别标签列表
        class_probabilities: 类别概率分布列表
    """
    cluster_classes = []
    class_probabilities = []

    for i, cluster_points in enumerate(clusters_coords):
        try:
            # 提取特征
            bbox = compute_bbox(cluster_points)
            x, y, z, width, length, height, _ = bbox
            volume = width * length * height
            n_points = len(cluster_points)
            density = n_points / volume if volume > 0.0001 else 0
            length = max(length, width)
            width = min(length, width)
            # 计算比率
            aspect_ratio = length /(width + 1e-6)
            height_ratio = height / (length + 1e-6)

            # 初始化效用分数
            utility_scores = np.zeros(4)  # [人, 小型车, 大型车, 其他]

            # ======== 人的效用分数 ========
            if 1 <= height <= 2.0:
                utility_scores[0] += 2
            if width <= 0.8 and length <= 0.8:
                utility_scores[0] += 3
            if height_ratio >= 1.5:
                utility_scores[0] += 3
            if volume < 1.0:
                utility_scores[0] += 2
            if density >= 10:
                utility_scores[0] += 1
            else:
                utility_scores[0] -= 2
            if 0.5<=z < 1.5:
                utility_scores[0] += 1
            else:
                utility_scores[0] -= 10

            # ======== 小型车效用分数 ========
            if 1.0 <= height <= 2.2:
                utility_scores[1] += 1
            if 1.5 <= width <= 2.5:
                utility_scores[1] += 2
            if 2.5 <= length <= 4:
                utility_scores[1] += 2
            if 1.5 <= aspect_ratio <= 4.0:
                utility_scores[1] += 2
            if 8<volume <= 40:
                utility_scores[1] += 3
            if density >= 10:
                utility_scores[1] += 1
            else:
                utility_scores[1] -= 2
            if 0.5<=z < 1.5:
                utility_scores[1] += 1
            else:
                utility_scores[1] -= 10

            # ======== 大型车效用分数 ========
            if 2.5 <= height <= 4.5:
                utility_scores[2] += 1
            if 2.0 <= width <= 3.0:
                utility_scores[2] += 2
            if 5.0 <= length <= 15.0:
                utility_scores[2] += 2
            if aspect_ratio >= 2.0:
                utility_scores[2] += 2
            if volume > 40.0:
                utility_scores[2] += 3
            if density >= 10:
                utility_scores[2] += 1
            else:
                utility_scores[2] -= 2
            if 1<= z < 2.0:
                utility_scores[2] += 1
            else:
                utility_scores[2] -= 10

            # 应用阈值以筛选明确的类别
            # 如果最高分的类别不是"其他"，且分数太低，则改为"其他"
            max_class = np.argmax(utility_scores)
            if utility_scores[max_class] < 6:
                utility_scores[3] += 12

            # 应用softmax获取概率
            temperature = 1.0  # 控制概率分布的锐度
            utility_scores = utility_scores / temperature
            # 为避免数值溢出，减去最大效用分数
            utility_scores = utility_scores - np.max(utility_scores)
            exp_scores = np.exp(utility_scores)
            probs = exp_scores / np.sum(exp_scores)

            # 获取预测类别（最高概率）
            class_label = np.argmax(probs)

            # 存储结果
            cluster_classes.append(class_label)
            class_probabilities.append(probs)

        except Exception as e:
            print(f"Error classifying cluster {i}: {e}")
            # 默认分类为"其他"
            cluster_classes.append(3)
            class_probabilities.append(np.array([0.0, 0.0, 0.0, 1.0]))
    if class_probabilities:
        class_probabilities = np.max(class_probabilities, axis=1)
    else:
        class_probabilities = np.array([])
    return cluster_classes, class_probabilities