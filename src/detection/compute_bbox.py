# import numpy as np
# from numba import jit, prange
#
# # @jit(nopython=True, cache=True)
# def compute_bbox(points_coords):
#     """计算具有最小xy平面投影面积的边界框，严格包络所有点"""
#     if len(points_coords) == 0:
#         # 处理空点集的情况
#         return [0, 0, 0, 0, 0, 0, 0]
#     min_coords = np.min(points_coords, axis=0)  # 计算点云的最小坐标
#     max_coords = np.max(points_coords, axis=0)  # 计算点云的最大坐标
#     centroid_coords = (min_coords + max_coords) / 2  # 计算点云的中心
#     # centroid_coords = np.mean(points_coords, axis=0)  # 计算点云的中心
#     centered_points_coords = points_coords - centroid_coords  # 将点云移动到中心
#     centered_points_2d_coords = centered_points_coords[:, :2]
#     best_yaw = 0.0
#     min_area = float('inf')
#     for angle_deg in np.arange(0, 90, 0.1):  # 遍历所有可能的旋转角度
#         angle = np.radians(angle_deg)
#         c, s = np.cos(angle), np.sin(angle)
#         R = np.array([[c, -s], [s, c]])
#         rotated_points_coords = np.dot(centered_points_2d_coords, R)
#         min_x, min_y = np.min(rotated_points_coords, axis=0)
#         max_x, max_y = np.max(rotated_points_coords, axis=0)
#         lower_x_near_edge = min_x + (max_x - min_x)/6
#         upper_x_near_edge = max_x + (max_x - min_x)*5/6
#         lower_y_near_edge = min_y + (max_y - min_y)/6
#         upper_y_near_edge = max_y + (max_y - min_y)*5/6
#         percent_in_center = sum(
#             lower_x_near_edge <= i[0] <= upper_x_near_edge and lower_y_near_edge <= i[1] <= upper_y_near_edge for i in
#             rotated_points_coords) / len(rotated_points_coords)
#         width = max_x - min_x
#         length = max_y - min_y
#         area = width * length
#         if percent_in_center >= 0.3:
#             area *= 2
#         if area < min_area:
#             min_area = area
#             best_yaw = angle
#     # 使用最优角度创建完整的3D边界框
#     c, s = np.cos(best_yaw), np.sin(best_yaw)
#     R_3d = np.array([
#         [c, -s, 0],
#         [s, c, 0],
#         [0, 0, 1]
#     ])
#     # 对所有点应用最优旋转
#     rotated_points_3d_coords = np.dot(centered_points_coords, R_3d)
#     # 计算旋转后的轴对齐边界框
#     rot_min_coords = np.min(rotated_points_3d_coords, axis=0)
#     rot_max_coords = np.max(rotated_points_3d_coords, axis=0)
#     # 精确计算边界框尺寸（确保没有额外的余量）
#     size = rot_max_coords - rot_min_coords
#     # 边界框中心（在旋转坐标系中）
#     bbox_center_rotated_coords = (rot_min_coords + rot_max_coords) / 2
#     # 将边界框中心转回原始坐标系
#     bbox_center_coords = centroid_coords + np.dot(bbox_center_rotated_coords, R_3d.T)
#     # 验证边界框是否完全包含所有点（使用极小的容差）
#     tolerance = 1e-6  # 极小的容差用于处理浮点精度问题
#     # 将所有点转换到边界框坐标系
#     points_bbox_coords = np.dot(points_coords - bbox_center_coords, R_3d)
#     # 检查所有点是否在边界框内
#     half_size = size / 2
#     points_outside = np.any(np.abs(points_bbox_coords) > half_size + tolerance, axis=1)
#     # 如果有点在边界框外，精确扩展边界框
#     if np.any(points_outside):
#         # 对每个维度分别计算需要的扩展量
#         min_vals = np.min(points_bbox_coords, axis=0)
#         max_vals = np.max(points_bbox_coords, axis=0)
#         # 计算每个维度需要的调整
#         size_adjustment = np.zeros(3)
#         for i in range(3):
#             if min_vals[i] < -half_size[i] - tolerance:
#                 size_adjustment[i] += 2 * ((-half_size[i] - tolerance) - min_vals[i])
#             if max_vals[i] > half_size[i] + tolerance:
#                 size_adjustment[i] += 2 * (max_vals[i] - (half_size[i] + tolerance))
#         # 更新尺寸
#         size += size_adjustment
#     # 最终的边界框参数 [x,y,z,w,l,h,yaw]
#     bbox = [
#         bbox_center_coords[0],
#         bbox_center_coords[1],
#         bbox_center_coords[2],
#         size[0],  # width
#         size[1],  # length
#         size[2],  # height
#         best_yaw
#     ]
#     return bbox

import numpy as np
from numba import jit, prange

# @jit(nopython=True, cache=True)
def compute_bbox(points_coords):
    """计算具有最小xy平面投影面积的边界框，严格包络所有点"""
    if len(points_coords) == 0:
        # 处理空点集的情况
        return [0, 0, 0, 0, 0, 0, 0]
    min_coords = np.min(points_coords, axis=0)  # 计算点云的最小坐标
    max_coords = np.max(points_coords, axis=0)  # 计算点云的最大坐标
    centroid_coords = (min_coords + max_coords) / 2  # 计算点云的中心
    # centroid_coords = np.mean(points_coords, axis=0)  # 计算点云的中心
    centered_points_coords = points_coords - centroid_coords  # 将点云移动到中心
    centered_points_2d_coords = centered_points_coords[:, :2]
    best_yaw = 0.0
    min_area = float('inf')
    for angle_deg in np.arange(0, 90, 0.1):  # 遍历所有可能的旋转角度
        angle = np.radians(angle_deg)
        c, s = np.cos(angle), np.sin(angle)
        R = np.array([[c, -s], [s, c]])
        rotated_points_coords = np.dot(centered_points_2d_coords, R)
        min_x, min_y = np.min(rotated_points_coords, axis=0)
        max_x, max_y = np.max(rotated_points_coords, axis=0)
        width = max_x - min_x
        length = max_y - min_y
        area = width * length
        if area < min_area:
            min_area = area
            best_yaw = angle
    # 使用最优角度创建完整的3D边界框
    c, s = np.cos(best_yaw), np.sin(best_yaw)
    R_3d = np.array([
        [c, -s, 0],
        [s, c, 0],
        [0, 0, 1]
    ])
    # 对所有点应用最优旋转
    rotated_points_3d_coords = np.dot(centered_points_coords, R_3d)
    # 计算旋转后的轴对齐边界框
    rot_min_coords = np.min(rotated_points_3d_coords, axis=0)
    rot_max_coords = np.max(rotated_points_3d_coords, axis=0)
    # 精确计算边界框尺寸（确保没有额外的余量）
    size = rot_max_coords - rot_min_coords
    # 边界框中心（在旋转坐标系中）
    bbox_center_rotated_coords = (rot_min_coords + rot_max_coords) / 2
    # 将边界框中心转回原始坐标系
    bbox_center_coords = centroid_coords + np.dot(bbox_center_rotated_coords, R_3d.T)
    # 验证边界框是否完全包含所有点（使用极小的容差）
    tolerance = 1e-6  # 极小的容差用于处理浮点精度问题
    # 将所有点转换到边界框坐标系
    points_bbox_coords = np.dot(points_coords - bbox_center_coords, R_3d)
    # 检查所有点是否在边界框内
    half_size = size / 2
    points_outside = np.any(np.abs(points_bbox_coords) > half_size + tolerance, axis=1)
    # 如果有点在边界框外，精确扩展边界框
    if np.any(points_outside):
        # 对每个维度分别计算需要的扩展量
        min_vals = np.min(points_bbox_coords, axis=0)
        max_vals = np.max(points_bbox_coords, axis=0)
        # 计算每个维度需要的调整
        size_adjustment = np.zeros(3)
        for i in range(3):
            if min_vals[i] < -half_size[i] - tolerance:
                size_adjustment[i] += 2 * ((-half_size[i] - tolerance) - min_vals[i])
            if max_vals[i] > half_size[i] + tolerance:
                size_adjustment[i] += 2 * (max_vals[i] - (half_size[i] + tolerance))
        # 更新尺寸
        size += size_adjustment
    # 最终的边界框参数 [x,y,z,w,l,h,yaw]
    bbox = [
        bbox_center_coords[0],
        bbox_center_coords[1],
        bbox_center_coords[2],
        size[0],  # width
        size[1],  # length
        size[2],  # height
        best_yaw
    ]
    return bbox