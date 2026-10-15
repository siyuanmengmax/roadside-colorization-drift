import numpy as np
from scipy.spatial import cKDTree
import hdbscan
from numba import jit, prange

from .compute_bbox import compute_bbox

# @jit(nopython=True, cache=True)
def clustering_stage_1(points, columns, rho_min=10, alpha=0.5, eps_min=0.5, eps_max=1,density_unit=1):
    n_points = len(points)  # 点的数量
    # 提取点的坐标, x,y,z,range,reflectivity,near_ir,intensity,flags,dynamic_mask,point_reliability,lidar_source
    point_coord = points[:, :columns.index('z')+1]  # 提取点的坐标
    tree = cKDTree(point_coord)  # KD树用于快速邻近点查找
    point_reliability = points[:, columns.index('reliability')]  # 提取点的可靠度
    basic_point_neighbor_id = tree.query_ball_point(point_coord, r=eps_max) # 获取所有点的基础邻居索引
    basic_point_density = np.array([density_unit * np.sum(point_reliability[ids]) for ids in basic_point_neighbor_id])
    min_basic_point_density = np.min(basic_point_density) # 计算最小基础点密度
    max_basic_point_density = np.max(basic_point_density) # 计算最大基础点密度
    mean_basic_point_density = np.mean(basic_point_density) # 计算平均基础点密度
    # print(f"最小基础点密度: {min_basic_point_density}, 最大基础点密度: {max_basic_point_density}, 平均基础点密度: {mean_basic_point_density}")
    utility = 1 - np.clip(rho_min / basic_point_density, 0, 1)  # 计算效用函数
    adaptive_radius = np.clip(eps_max * (1.0 - alpha * utility), eps_min, eps_max) # 计算自适应半径，使用线性插值
    # 聚类过程
    clusters = [] # 存储聚类结果，包含每个聚类的点id列表
    clusters_coord = []  # 存储聚类的点坐标
    # processed_point_id = set() # 存储已处理的点id
    processed = np.zeros(n_points, dtype=bool)
    for i in range(n_points): # 遍历所有点
        # if i not in processed_point_id: # 如果点已处理，跳过
        if not processed[i]: # 如果点已处理，跳过
            adaptive_point_neighbor_id = tree.query_ball_point(point_coord[i], r=adaptive_radius[i]) # 获取自适应邻居索引
            adaptive_point_density = density_unit * np.sum(point_reliability[adaptive_point_neighbor_id]) # 计算自适应点密度
            if adaptive_point_density >= rho_min: # 如果密度满足条件，开始扩展聚类
                cluster_point_id = [] # 存储当前聚类的点id
                points_to_process = list(adaptive_point_neighbor_id)
                cluster_point_id.append(i) # 将当前点添加到聚类中
                # processed_point_id.add(i) # 标记当前点为已处理
                processed[i] = True
                while points_to_process: # 处理待处理点
                    j = points_to_process.pop() # 获取待处理点id
                    # if j not in processed_point_id:  # 如果点已处理，跳过
                    if not processed[j]: # 如果点已处理，跳过
                        # processed_point_id.add(j) # 标记当前点为已处理
                        processed[j] = True
                        new_adaptive_point_neighbor_id = tree.query_ball_point(point_coord[j], r=adaptive_radius[j]) # 获取新邻居索引
                        new_adaptive_point_density = np.sum(point_reliability[new_adaptive_point_neighbor_id]) # 计算新邻居的密度
                        if new_adaptive_point_density >= rho_min: # 如果新邻居密度满足条件，添加到待处理列表
                            # new_points = set(new_adaptive_point_neighbor_id) -processed_point_id
                            new_points = [j for j in new_adaptive_point_neighbor_id if not processed[j]]
                            points_to_process.extend(new_points)
                        cluster_point_id.append(j) # 将当前点添加到聚类中
                clusters.append(cluster_point_id) # 将当前聚类的点id添加到聚类列表
                clusters_coord.append(point_coord[cluster_point_id]) # 将当前聚类的点坐标添加到聚类坐标列表
    return clusters, clusters_coord, adaptive_radius # 返回聚类结果，包含每个聚类的点id列表、聚类的点坐标列表、自适应半径

def clustering_stage_1_no_rel(points, columns, rho_min=10, alpha=0.5, eps_min=0.5, eps_max=1,density_unit=1):
    n_points = len(points)  # 点的数量
    # 提取点的坐标, x,y,z,range,reflectivity,near_ir,intensity,flags,dynamic_mask,point_reliability,lidar_source
    point_coord = points[:, :columns.index('z')+1]  # 提取点的坐标
    tree = cKDTree(point_coord)  # KD树用于快速邻近点查找
    # 点的可靠度全部设为1
    point_reliability = np.ones(n_points)
    basic_point_neighbor_id = tree.query_ball_point(point_coord, r=eps_max) # 获取所有点的基础邻居索引
    basic_point_density = np.array([density_unit * np.sum(point_reliability[ids]) for ids in basic_point_neighbor_id])
    min_basic_point_density = np.min(basic_point_density) # 计算最小基础点密度
    max_basic_point_density = np.max(basic_point_density) # 计算最大基础点密度
    mean_basic_point_density = np.mean(basic_point_density) # 计算平均基础点密度
    # print(f"最小基础点密度: {min_basic_point_density}, 最大基础点密度: {max_basic_point_density}, 平均基础点密度: {mean_basic_point_density}")
    utility = 1 - np.clip(rho_min / basic_point_density, 0, 1)  # 计算效用函数
    adaptive_radius = np.clip(eps_max * (1.0 - alpha * utility), eps_min, eps_max) # 计算自适应半径，使用线性插值
    # 聚类过程
    clusters = [] # 存储聚类结果，包含每个聚类的点id列表
    clusters_coord = []  # 存储聚类的点坐标
    # processed_point_id = set() # 存储已处理的点id
    processed = np.zeros(n_points, dtype=bool)
    for i in range(n_points): # 遍历所有点
        # if i not in processed_point_id: # 如果点已处理，跳过
        if not processed[i]: # 如果点已处理，跳过
            adaptive_point_neighbor_id = tree.query_ball_point(point_coord[i], r=adaptive_radius[i]) # 获取自适应邻居索引
            adaptive_point_density = density_unit * np.sum(point_reliability[adaptive_point_neighbor_id]) # 计算自适应点密度
            if adaptive_point_density >= rho_min: # 如果密度满足条件，开始扩展聚类
                cluster_point_id = [] # 存储当前聚类的点id
                points_to_process = list(adaptive_point_neighbor_id)
                cluster_point_id.append(i) # 将当前点添加到聚类中
                # processed_point_id.add(i) # 标记当前点为已处理
                processed[i] = True
                while points_to_process: # 处理待处理点
                    j = points_to_process.pop() # 获取待处理点id
                    # if j not in processed_point_id:  # 如果点已处理，跳过
                    if not processed[j]: # 如果点已处理，跳过
                        # processed_point_id.add(j) # 标记当前点为已处理
                        processed[j] = True
                        new_adaptive_point_neighbor_id = tree.query_ball_point(point_coord[j], r=adaptive_radius[j]) # 获取新邻居索引
                        new_adaptive_point_density = np.sum(point_reliability[new_adaptive_point_neighbor_id]) # 计算新邻居的密度
                        if new_adaptive_point_density >= rho_min: # 如果新邻居密度满足条件，添加到待处理列表
                            # new_points = set(new_adaptive_point_neighbor_id) -processed_point_id
                            new_points = [j for j in new_adaptive_point_neighbor_id if not processed[j]]
                            points_to_process.extend(new_points)
                        cluster_point_id.append(j) # 将当前点添加到聚类中
                clusters.append(cluster_point_id) # 将当前聚类的点id添加到聚类列表
                clusters_coord.append(point_coord[cluster_point_id]) # 将当前聚类的点坐标添加到聚类坐标列表
    return clusters, clusters_coord, adaptive_radius # 返回聚类结果，包含每个聚类的点id列表、聚类的点坐标列表、自适应半径

def clustering_stage_1_color(points, columns, rgb_colors, rgb_rel,
                              rho_min=10, alpha=0.5, eps_min=0.5, eps_max=1,
                              delta=30.0, rgb_gate=0.25, density_unit=1):
    """Quality-aware adaptive DBSCAN with color-consistency gating.

    Extends clustering_stage_1 with a per-point color constraint: during
    cluster expansion, candidate point j is only admitted if its mean-L1
    color distance from the growing cluster's running-mean color is below
    delta — but ONLY when both j and the cluster seed have rgb_rel > rgb_gate.
    Below rgb_gate (e.g. night, rgb_rel=0.116) the check is skipped and the
    algorithm reduces to standard quality-aware DBSCAN.

    Args:
        rgb_colors: (N, 3) float32, per-point RGB 0-255 (from born_fused_360).
        rgb_rel:    (N,)   float32, per-point spectral reliability (0-1).
        delta:      max mean-L1 color distance for cluster membership (RGB units).
        rgb_gate:   min rgb_rel required to activate color constraint.
    """
    n_points = len(points)
    point_coord = points[:, :columns.index('z') + 1]
    tree = cKDTree(point_coord)
    point_reliability = points[:, columns.index('reliability')]

    basic_neighbor_ids = tree.query_ball_point(point_coord, r=eps_max)
    basic_density = np.array([density_unit * np.sum(point_reliability[ids])
                               for ids in basic_neighbor_ids])
    utility = 1 - np.clip(rho_min / basic_density, 0, 1)
    adaptive_radius = np.clip(eps_max * (1.0 - alpha * utility), eps_min, eps_max)

    clusters, clusters_coord = [], []
    processed = np.zeros(n_points, dtype=bool)

    for i in range(n_points):
        if processed[i]:
            continue
        neighbors = tree.query_ball_point(point_coord[i], r=adaptive_radius[i])
        density = density_unit * np.sum(point_reliability[neighbors])
        if density < rho_min:
            continue

        # Seed this cluster; initialise running color mean from seed point
        cluster_ids = []
        use_color_i = rgb_rel[i] > rgb_gate
        color_sum = rgb_colors[i].astype(np.float64).copy()
        color_n = 1

        to_process = list(neighbors)
        cluster_ids.append(i)
        processed[i] = True

        while to_process:
            j = to_process.pop()
            if processed[j]:
                continue
            processed[j] = True

            # Color consistency gate (only when both seed and j have reliable RGB)
            if use_color_i and rgb_rel[j] > rgb_gate:
                cluster_mean_color = color_sum / color_n
                color_dist = np.mean(np.abs(rgb_colors[j].astype(np.float64)
                                            - cluster_mean_color))
                if color_dist > delta:
                    continue  # reject: color too different from cluster

            new_neighbors = tree.query_ball_point(point_coord[j], r=adaptive_radius[j])
            new_density = np.sum(point_reliability[new_neighbors])
            if new_density >= rho_min:
                to_process.extend([k for k in new_neighbors if not processed[k]])

            cluster_ids.append(j)
            # Update running color mean
            color_sum += rgb_colors[j].astype(np.float64)
            color_n += 1

        clusters.append(cluster_ids)
        clusters_coord.append(point_coord[cluster_ids])

    return clusters, clusters_coord, adaptive_radius


# @jit(nopython=True, cache=True)
def clustering_stage_2(clusters, clusters_coord, cluster_classes, points, columns):
    # Define parameters for each coarse class
    params = {'max_volume': 120,
              'min_aspect_ratio': 1.5,
              'max_aspect_ratio': 5,
              'min_distance': 1.5
              }
    while True:
        n_clusters = len(clusters)
        if n_clusters <= 1:
            break
        merge_candidates = []  # Store all potential merge candidates
        # Calculate distances between all cluster pairs
        for i in range(n_clusters):
            for j in range(i + 1, n_clusters):
                # Get class-specific parameters
                if cluster_classes[i] == 0 and cluster_classes[j] == 0:
                    continue
                class_id = cluster_classes[i]
                min_distance = params['min_distance']
                max_volume = params['max_volume']
                min_aspect_ratio = params['min_aspect_ratio']
                max_aspect_ratio = params['max_aspect_ratio']
                # Calculate minimum distance between clusters
                if len(clusters_coord[i]) < len(clusters_coord[j]):
                    tree = cKDTree(clusters_coord[i])
                    dist, _ = tree.query(clusters_coord[j], k=1)
                else:
                    tree = cKDTree(clusters_coord[j])
                    dist, _ = tree.query(clusters_coord[i], k=1)
                min_dist = np.min(dist)
                if min_dist <= min_distance:
                    # Merge clusters temporarily
                    merged_points = clusters[i] + clusters[j]
                    merged_coords = np.row_stack([clusters_coord[i], clusters_coord[j]])
                    # Calculate bounding box and constraints
                    bbox = compute_bbox(merged_coords)
                    volume = bbox[3] * bbox[4] * bbox[5]
                    if volume <= max_volume:
                        width = min(bbox[3], bbox[4])
                        length = max(bbox[3], bbox[4])
                        if width > 0:
                            aspect_ratio = length / width
                            if min_aspect_ratio <= aspect_ratio <= max_aspect_ratio:
                                merge_candidates.append({
                                    'distance': min_dist,
                                    'i': i,
                                    'j': j,
                                    'merged_points': merged_points,
                                    'merged_coords': merged_coords,
                                    'class_id': class_id
                                })
        # If no candidates found, stop merging
        if not merge_candidates:
            break
        # Sort candidates by distance
        merge_candidates.sort(key=lambda x: x['distance'])
        # Greedy merging strategy
        merged_cluster_ids = set()
        new_clusters = []
        new_clusters_coord = []
        new_cluster_classes = []
        for candidate in merge_candidates:
            i, j = candidate['i'], candidate['j']
            if i not in merged_cluster_ids and j not in merged_cluster_ids:
                # Merge clusters
                new_clusters.append(candidate['merged_points'])
                new_clusters_coord.append(candidate['merged_coords'])
                new_cluster_classes.append(candidate['class_id'])
                merged_cluster_ids.add(i)
                merged_cluster_ids.add(j)
        # Add unmerged clusters
        for i in range(n_clusters):
            if i not in merged_cluster_ids:
                new_clusters.append(clusters[i])
                new_clusters_coord.append(clusters_coord[i])
                new_cluster_classes.append(cluster_classes[i])
        # If no merges happened, stop
        if len(new_clusters) == n_clusters:
            break
        # Update for next iteration
        clusters = new_clusters
        clusters_coord = new_clusters_coord
        cluster_classes = new_cluster_classes
    return clusters, clusters_coord, cluster_classes

def dbscan_clustering(points, columns, eps=1, min_points=50):
    """
    基于DBSCAN算法的点云聚类

    参数:
        points: 点云数据, numpy数组, 包含所有点的坐标和其他属性
        columns: 列名称列表, 用于识别x,y,z等字段的位置
        eps: DBSCAN的邻域半径参数, 默认0.5
        min_points: DBSCAN的最小点数参数, 默认5

    返回:
        clusters: 列表, 每个元素是一个聚类的点索引列表
        clusters_coord: 列表, 每个元素是一个聚类的点坐标
        point_labels: numpy数组, 每个点的聚类标签
    """
    n_points = len(points)  # 点的数量
    # 提取点的坐标
    point_coord = points[:, :columns.index('z') + 1]  # 提取点的x,y,z坐标
    # 创建KD树用于快速邻近点查找
    tree = cKDTree(point_coord)
    # 初始化标签数组, -1表示噪声点, 0表示未处理点, 正数表示聚类ID
    labels = np.zeros(n_points, dtype=int) - 1
    # 聚类ID从1开始
    cluster_id = 0
    # 处理每个点
    for i in range(n_points):
        # 如果点已处理(已分配聚类ID或被标记为噪声), 跳过
        if labels[i] != -1:
            continue
        # 获取当前点的邻域内的点索引
        neighbor_indices = tree.query_ball_point(point_coord[i], r=eps)
        # 如果邻域内点数不足, 标记为噪声点
        if len(neighbor_indices) < min_points:
            labels[i] = -1
            continue
        # 创建新的聚类
        cluster_id += 1
        labels[i] = cluster_id
        # 使用队列进行区域扩展
        seeds = list(neighbor_indices)
        seeds.remove(i)  # 移除当前点
        # 扩展聚类
        j = 0
        while j < len(seeds):
            # 获取当前种子点
            current_point = seeds[j]
            # 如果点是噪声点, 将其添加到当前聚类
            if labels[current_point] == -1:
                labels[current_point] = cluster_id
            # 如果点未处理, 将其添加到当前聚类并检查其邻域
            elif labels[current_point] == 0:
                labels[current_point] = cluster_id
                # 获取当前种子点的邻域内的点
                current_neighbors = tree.query_ball_point(point_coord[current_point], r=eps)
                # 如果邻域足够密集, 将未处理的邻居添加到种子列表
                if len(current_neighbors) >= min_points:
                    for neighbor in current_neighbors:
                        if labels[neighbor] == 0 or labels[neighbor] == -1:
                            if neighbor not in seeds:
                                seeds.append(neighbor)
            j += 1
    # 处理结果并构建返回格式
    clusters = []  # 存储聚类结果, 包含每个聚类的点id列表
    clusters_coord = []  # 存储聚类的点坐标
    # 收集每个聚类的点
    for c in range(1, cluster_id + 1):
        cluster_indices = np.where(labels == c)[0]
        if len(cluster_indices) > 0:  # 确保聚类非空
            clusters.append(cluster_indices.tolist())
            clusters_coord.append(point_coord[cluster_indices])
    eps = np.full(n_points, eps)  # 返回每个点的eps值
    return clusters, clusters_coord, eps


def hdbscan_clustering(points, columns, min_cluster_size=50, min_samples=None,
                       cluster_selection_epsilon=0.5, alpha=1.0):
    """
    基于HDBSCAN算法的点云聚类
    参数:
        points: 点云数据, numpy数组, 包含所有点的坐标和其他属性
        columns: 列名称列表, 用于识别x,y,z等字段的位置
        min_cluster_size: 聚类的最小大小, 默认50
        min_samples: 核心点的最小邻居数, 默认None(自动设为min_cluster_size)
        cluster_selection_epsilon: 聚类选择的epsilon值, 默认0.0(使用层次化选择)
        alpha: 距离度量的alpha参数, 默认1.0
    返回:
        clusters: 列表, 每个元素是一个聚类的点索引列表
        clusters_coord: 列表, 每个元素是一个聚类的点坐标
        point_labels: numpy数组, 每个点的聚类标签
    """
    n_points = len(points)

    # 提取点的坐标
    point_coord = points[:, :columns.index('z') + 1]  # 提取点的x,y,z坐标

    # 设置默认参数
    if min_samples is None:
        min_samples = min_cluster_size

    # 创建HDBSCAN聚类器
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_epsilon=cluster_selection_epsilon,
        alpha=alpha,
        metric='euclidean',
        algorithm='best',
        leaf_size=40,
        # n_jobs=-1  # 使用所有CPU核心
    )

    # 执行聚类
    cluster_labels = clusterer.fit_predict(point_coord)

    # 处理结果并构建返回格式
    clusters = []  # 存储聚类结果, 包含每个聚类的点id列表
    clusters_coord = []  # 存储聚类的点坐标

    # 获取独特的聚类标签 (排除噪声点 -1)
    unique_labels = np.unique(cluster_labels)
    unique_labels = unique_labels[unique_labels >= 0]  # 过滤掉噪声标签-1

    # 收集每个聚类的点
    for label in unique_labels:
        cluster_indices = np.where(cluster_labels == label)[0]
        if len(cluster_indices) > 0:  # 确保聚类非空
            clusters.append(cluster_indices.tolist())
            clusters_coord.append(point_coord[cluster_indices])

    # 计算每个点的局部密度(用于compatibility)
    # 使用聚类器的outlier_scores_作为密度的倒数
    if hasattr(clusterer, 'outlier_scores_'):
        # outlier_scores_越高表示越像噪声,我们用1-score作为密度指标
        density_scores = 1.0 - clusterer.outlier_scores_
        density_scores = np.maximum(density_scores, 0.1)  # 确保最小密度
    else:
        # 如果没有outlier_scores,用固定值
        density_scores = np.full(n_points, 0.5)

    return clusters, clusters_coord, density_scores