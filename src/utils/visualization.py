# visualization.py

import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time
from glob import glob
import os
from tqdm import tqdm
import cv2
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from .io import create_pcd,read_bin

def visualize_pcd(final_pcd_array):
    """
    Visualize the final point cloud in gray color.
    Args:
        final_pcd (o3d.geometry.PointCloud): The point cloud to visualize
    """
    # Create a point cloud object
    final_pcd = create_pcd(final_pcd_array[:, :3])
    final_pcd.paint_uniform_color([0.5, 0.5, 0.5])  # Gray color
    # Create a visualizer object
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(final_pcd)
    # Set the rendering options
    opt = vis.get_render_option()
    opt.background_color = np.array([1, 1, 1])  # White background
    opt.point_size = 1.0  # Set point size
    # Run the visualizer
    vis.run()
    vis.destroy_window()

def visualize_pcd_with_bbox_3d(pcd, bboxes, z_min=-0.1, z_max=0.1,ground=False):
    """
    Visualize point cloud with 3D bounding boxes.
    :param pcd:
    :param bboxes:
    :param z_min:
    :param z_max:
    :return:
    """
    # Get points and colors
    points = np.asarray(pcd.points)
    colors = np.full_like(points, [0.5, 0.5, 0.5])  # Gray for point cloud
    # Filter ground points
    if ground:
        ground_points_mask = (points[:, 2] >= z_min) & (points[:, 2] <= z_max)
        colors[ground_points_mask] = [0, 0, 1]  # Blue for ground points
    pcd.colors = o3d.utility.Vector3dVector(colors)
    # Create a visualizer object
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    # Add bounding boxes
    for bbox in bboxes:
        center = bbox[:3]
        extent = bbox[3:6]
        R = o3d.geometry.get_rotation_matrix_from_xyz((0, 0, bbox[6]))
        obb = o3d.geometry.OrientedBoundingBox(center, R, extent)
        obb.color = (1, 0, 0)  # Red color for bounding box
        vis.add_geometry(obb)
    # Set the rendering options
    opt = vis.get_render_option()
    opt.background_color = np.array([1, 1, 1])  # 白色背景
    opt.point_size = 1.0
    # Add coordinate frame
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
    vis.add_geometry(coordinate_frame)
    # Run the visualizer
    vis.run()
    vis.destroy_window()

def visualize_pcd_with_bbox_3d_new(points, bboxes, z_min=-0.1, z_max=0.1,ground=False):
    """
    Visualize point cloud with 3D bounding boxes.
    :param pcd:
    :param bboxes:
    :param z_min:
    :param z_max:
    :return:
    """
    # Get points and colors
    pcd = create_pcd(points)
    colors = np.full_like(points, [0.5, 0.5, 0.5])  # Gray for point cloud
    # Filter ground points
    if ground:
        ground_points_mask = (points[:, 2] >= z_min) & (points[:, 2] <= z_max)
        colors[ground_points_mask] = [0, 0, 1]  # Blue for ground points
    pcd.colors = o3d.utility.Vector3dVector(colors)
    # Create a visualizer object
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    # Add bounding boxes
    for bbox in bboxes:
        center = bbox[:3]
        extent = bbox[3:6]
        R = o3d.geometry.get_rotation_matrix_from_xyz((0, 0, bbox[6]))
        obb = o3d.geometry.OrientedBoundingBox(center, R, extent)
        obb.color = (1, 0, 0)  # Red color for bounding box
        vis.add_geometry(obb)
    # Set the rendering options
    opt = vis.get_render_option()
    opt.background_color = np.array([1, 1, 1])  # 白色背景
    opt.point_size = 1.0
    # Add coordinate frame
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
    vis.add_geometry(coordinate_frame)
    # Run the visualizer
    vis.run()
    vis.destroy_window()
def visualize_pcd_with_bbox_2d(pcd, bboxes):
    """
    Visualize point cloud with 2D bounding boxes.
    :param pcd:
    :param bboxes:
    :return:
    """
    # Get points
    points = np.asarray(pcd.points)
    points_2d = points[:, :2]
    # Plot the point cloud
    plt.scatter(points_2d[:, 0], points_2d[:, 1], c='gray', s=1)
    # Create a visualizer object
    for bbox in bboxes:
        x, y, w, l, yaw = bbox
        # Calculate the 4 corners of the bounding box
        corners = np.array([
            [-w / 2, -l / 2],
            [w / 2, -l / 2],
            [w / 2, l / 2],
            [-w / 2, l / 2]
        ])
        # Rotate the corners
        rotation = np.array([
            [np.cos(yaw), -np.sin(yaw)],
            [np.sin(yaw), np.cos(yaw)]
        ])
        corners = np.dot(corners, rotation.T)
        # Translate the corners
        corners += np.array([x, y])
        # Plot the bounding box
        corners = np.vstack([corners, corners[0]])  # Close the polygon
        plt.plot(corners[:, 0], corners[:, 1], 'r-')
        # Plot the yaw arrow
        arrow_length = min(w, l) / 2
        plt.arrow(x, y, arrow_length * np.cos(yaw),
                  arrow_length * np.sin(yaw),
                  head_width=0.1, color='g')
    plt.axis('equal')
    plt.show()

def visualize_ground(ground_points, non_ground_points):
    """
    Visualize ground and non-ground points.
    :param ground_points:
    :param non_ground_points:
    :return:
    """
    ground_points.paint_uniform_color([1, 0, 0])  # Red for ground
    non_ground_points.paint_uniform_color([0.5, 0.5, 0.5])  # Gray for non-ground
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(ground_points)
    vis.add_geometry(non_ground_points)
    opt = vis.get_render_option()
    opt.background_color = np.array([1, 1, 1])
    opt.point_size = 1.0
    vis.run()
    vis.destroy_window()

def visualize_dynamic_points(dynamic_points_array, static_points_array):
    """
    Visualize dynamic and static points.
    :param dynamic_points:
    :param static_points:
    :return:
    """
    dynamic_points = create_pcd(dynamic_points_array[:, :3])
    static_points = create_pcd(static_points_array[:, :3])
    dynamic_points.paint_uniform_color([1, 0, 0])  # Red for dynamic points
    static_points.paint_uniform_color([0.5, 0.5, 0.5])  # Gray for static points
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(dynamic_points)
    vis.add_geometry(static_points)
    opt = vis.get_render_option()
    opt.background_color = np.array([1, 1, 1])
    opt.point_size = 1.0
    vis.run()
    vis.destroy_window()

def visualize_pcd_with_bbox_and_clusters(pcd, core_points, core_labels, bboxes, z_min=-0.1, z_max=0.1, ground=False):
    """
    Visualize point cloud with bounding boxes and clusters.
    :param pcd:
    :param core_points:
    :param core_labels:
    :param bboxes:
    :param z_min:
    :param z_max:
    :param ground:
    :return:
    """
    # Create a visualizer object
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    # 将原始点云的颜色设置为浅灰色且半透明
    orig_points = np.asarray(pcd.points)
    orig_colors = np.full_like(orig_points, [0.5, 0.5, 0.5])  # 灰色
    pcd.colors = o3d.utility.Vector3dVector(orig_colors)
    vis.add_geometry(pcd)
    # 为每个cluster创建单独的点云，并使用鲜艳的颜色
    unique_labels = np.unique(core_labels)
    bright_colors = [
        [1, 0, 0],  # 红色
        [0, 1, 0],  # 绿色
        [0, 0, 1],  # 蓝色
        [1, 1, 0],  # 黄色
        [1, 0, 1],  # 紫色
        [0, 1, 1],  # 青色
        [1, 0.5, 0],  # 橙色
        [0.5, 0, 1],  # 紫罗兰
        [0, 1, 0.5],  # 浅绿色
    ]
    for i, label in enumerate(unique_labels):
        # 获取当前cluster的点
        cluster_mask = core_labels == label
        cluster_points = core_points[cluster_mask]
        # 创建新的点云对象
        cluster_pcd = o3d.geometry.PointCloud()
        cluster_pcd.points = o3d.utility.Vector3dVector(cluster_points)
        # 设置鲜艳的颜色
        color = bright_colors[i % len(bright_colors)]
        cluster_colors = np.tile(color, (len(cluster_points), 1))
        cluster_pcd.colors = o3d.utility.Vector3dVector(cluster_colors)
        # 增大点的大小（通过复制近邻点的方式）
        enlarged_points = []
        enlarged_colors = []
        for p, c in zip(cluster_points, cluster_colors):
            enlarged_points.extend([p + np.array([dx, dy, dz]) * 0.02
                                    for dx in [-1, 0, 1]
                                    for dy in [-1, 0, 1]
                                    for dz in [-1, 0, 1]])
            enlarged_colors.extend([c] * 27)
        enlarged_pcd = o3d.geometry.PointCloud()
        enlarged_pcd.points = o3d.utility.Vector3dVector(enlarged_points)
        enlarged_pcd.colors = o3d.utility.Vector3dVector(enlarged_colors)
        vis.add_geometry(enlarged_pcd)
    # 添加bounding boxes
    for i, bbox in enumerate(bboxes):
        center = bbox[:3]
        extent = bbox[3:6]
        R = o3d.geometry.get_rotation_matrix_from_xyz((0, 0, bbox[6]))
        obb = o3d.geometry.OrientedBoundingBox(center, R, extent)
        # 使用和cluster相同的颜色
        color = bright_colors[i % len(bright_colors)]
        obb.color = color
        vis.add_geometry(obb)
    # Set the rendering options
    opt = vis.get_render_option()
    opt.background_color = np.array([1, 1, 1])  # 白色背景
    opt.point_size = 1.0  # 增大点的大小
    # Add coordinate frame
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
    vis.add_geometry(coordinate_frame)
    # Set view control for better visualization
    vc = vis.get_view_control()
    vc.set_front([0, -1, -0.5])
    vc.set_lookat(np.mean(core_points, axis=0))
    vc.set_up([0, 0, 1])
    vc.set_zoom(0.7)
    # Run the visualizer
    vis.run()
    vis.destroy_window()

def visualize_tracking_2d_static(tracked_objects, frequency, km_per_h=True, arrow=False, interval=1):
    """
    Visualize the tracking results.
    :param tracked_objects:
    :param frame_id_mapping:
    :param frequency:
    :param km_per_h:
    :param arrow:
    :return:
    """
    if km_per_h:
        frequency = frequency * 3.6
    for frame_id, objects in enumerate(tracked_objects):
        if frame_id % interval == 0:
            fig, ax = plt.subplots(figsize=(12, 12))
            ax.set_aspect('equal')
            ax.clear()
            for obj in objects:
                bbox = obj['bbox']  # [x, y, w, l, yaw]
                # print(f'Frame {frame_id}, Track ID: {obj["track_id"]}, yaw: {np.rad2deg(bbox[6]):.1f}')
                track_id = obj['track_id']
                speed = obj['speed']
                speed_magnitude = np.linalg.norm(speed) * frequency
                # 使用bbox中的yaw创建旋转矩形
                rect = patches.Rectangle(
                    (bbox[0] - bbox[3] / 2, bbox[1] - bbox[4] / 2),
                    bbox[3], bbox[4],
                    angle=np.rad2deg(bbox[6]),
                    edgecolor='r',
                    facecolor='none',
                    alpha=1,  # 边框的透明度
                    linewidth=1.5,
                    rotation_point='center'
                )
                ax.add_patch(rect)
                if arrow:
                    arrow_length = 5.0  # 设置箭头长度
                    dx = arrow_length * np.cos(bbox[4])
                    dy = arrow_length * np.sin(bbox[4])
                    ax.arrow(bbox[0], bbox[1], dx, dy,
                             head_width=0.5, head_length=1.0, fc='r', ec='r')
                # 添加ID和速度标签
                ax.text(bbox[0]+bbox[3], bbox[1]+bbox[4], f'id:{track_id} v:{speed_magnitude:.1f}km/h', color='black')
            ax.set_xlim(-75, 75)
            ax.set_ylim(-75, 75)
            ax.grid(True)
            ax.set_title(f'Frame ID: {frame_id}')  # 使用原始frame_id
            plt.show()

def visualize_tracking_2d_dynamic(tracked_objects, frequency, output_path='output_2d.mp4', km_per_h=True,
                                  arrow=False, delay=0.001, interval=1,show=False):
    """
    Visualize the tracking results and save as video using OpenCV.
    """
    # 创建figure时设置背景色为白色
    fig, ax = plt.subplots(figsize=(12, 12), facecolor='white')
    ax.set_facecolor('white')
    ax.set_aspect('equal')
    # 设置视频写入器
    first_frame = True
    video_writer = None
    trajectories = {}
    labels = ['car','person']
    if km_per_h:
        frequency = frequency * 3.6
    try:
        for frame_id, objects in tqdm(enumerate(tracked_objects)):
            if frame_id % interval == 0:
                ax.clear()
                ax.set_facecolor('white')
                active_ids = []
                for obj in objects:
                    bbox = obj['bbox']
                    track_id = obj['track_id']
                    speed = obj['speed']
                    label = obj['label']
                    speed_magnitude = np.linalg.norm(speed) * frequency
                    active_ids.append(track_id)
                    # 绘制旋转矩形
                    rect = patches.Rectangle(
                        (bbox[0] - bbox[3] / 2, bbox[1] - bbox[4] / 2),
                        bbox[3], bbox[4],
                        angle=np.rad2deg(bbox[6]),
                        edgecolor='r',
                        facecolor='none',
                        alpha=1,
                        linewidth=1.5,
                        rotation_point='center'
                    )
                    ax.add_patch(rect)
                    if arrow:
                        arrow_length = 5.0
                        dx = arrow_length * np.cos(bbox[4])
                        dy = arrow_length * np.sin(bbox[4])
                        ax.arrow(bbox[0], bbox[1], dx, dy,
                                 head_width=0.5, head_length=1.0, fc='r', ec='r')
                    ax.text(bbox[0] + bbox[3], bbox[1] + bbox[4],
                            f'id:{track_id} v:{speed_magnitude:.1f}km/h, label:{labels[label]}', color='black')
                    if track_id not in trajectories:
                        trajectories[track_id] = []
                    trajectories[track_id].append([bbox[0], bbox[1]])
                    if track_id in trajectories:
                        trajectory = np.array(trajectories[track_id])
                        ax.plot(trajectory[:, 0], trajectory[:, 1],
                                '-', color='r', linewidth=1, alpha=0.5)
                trajectories = {k: v for k, v in trajectories.items() if k in active_ids}
                ax.set_xlim(-75, 75)
                ax.set_ylim(-75, 75)
                ax.grid(True)
                ax.set_title(f'Frame ID: {frame_id}')
                # 转换图形为RGB数组
                plt.tight_layout()
                fig.canvas.draw()
                buf = fig.canvas.buffer_rgba()
                w, h = fig.canvas.get_width_height()
                frame=np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4)
                # 转换RGBA为BGR格式
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
                if first_frame:
                    height, width = frame_bgr.shape[:2]
                    video_writer = cv2.VideoWriter(
                        output_path,
                        cv2.VideoWriter_fourcc(*'mp4v'),
                        20,  # FPS
                        (width, height)
                    )
                    first_frame = False
                video_writer.write(frame_bgr)
                if show and delay > 0:
                    plt.pause(delay)
    finally:
        if video_writer is not None:
            video_writer.release()
        plt.close()

def natural_sort_key(s):
    """
    用于实现自然排序的键函数
    将字符串中的数字部分转换为整数进行比较
    """
    import re
    def convert(text):
        return int(text) if text.isdigit() else text.lower()
    return [convert(c) for c in re.split('([0-9]+)', s)]


def compute_cluster_info(points, cluster_mask, source, cluster_param=None):
    """计算单个cluster的基本信息"""
    cluster_points = points[cluster_mask]

    # 基本统计信息
    num_points = len(cluster_points)

    # 计算点云密度
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cluster_points)
    bbox = pcd.get_oriented_bounding_box()
    volume = bbox.volume()
    density = num_points / volume if volume > 0 else 0

    # 计算来源占比
    cluster_source = source[cluster_mask]
    unique_sources, source_counts = np.unique(cluster_source, return_counts=True)
    source_ratios = {int(src): count / num_points for src, count in zip(unique_sources, source_counts)}

    info = {
        'num_points': num_points,
        'density': density,
        'source_ratios': source_ratios
    }

    # 添加聚类参数（如果有）
    if cluster_param is not None:
        info.update(cluster_param)

    return info


def format_cluster_text1(info, cluster_id):
    """格式化cluster信息文本"""
    text = f"ID: {cluster_id}\n"
    text += f"Points: {info['num_points']}\n"
    text += f"Density: {info['density']:.1f}\n"

    # 显示DBSCAN参数（如果有）
    if 'eps' in info:
        text += f"eps: {info['eps']:.2f}\n"
    if 'min_samples' in info:
        text += f"min_pts: {info['min_samples']}\n"
    if 'avg_distance' in info:
        text += f"avg_dist: {info['avg_distance']:.1f}\n"

    # 显示来源占比
    ratios = info['source_ratios']
    text += "Source: "
    if 0 in ratios and 1 in ratios:
        text += f"{ratios[0]:.1%}/{ratios[1]:.1%}"
    elif 0 in ratios:
        text += f"{ratios[0]:.1%}/0%"
    elif 1 in ratios:
        text += f"0%/{ratios[1]:.1%}"

    return text


def format_bbox_info(cluster_id, info):
    """格式化单行bbox信息"""
    text = f"ID {cluster_id}: "
    text += f"Points {info['num_points']}, "
    text += f"Density {info['density']:.1f}, "

    # 添加source ratio
    ratios = info['source_ratios']
    if 0 in ratios and 1 in ratios:
        text += f"Source {ratios[0]:.1%}/{ratios[1]:.1%}"
    elif 0 in ratios:
        text += f"Source {ratios[0]:.1%}/0%"
    elif 1 in ratios:
        text += f"Source 0%/{ratios[1]:.1%}"

    if 'eps' in info:
        text += f", eps {info['eps']:.2f}"
    if 'min_samples' in info:
        text += f", min_pts {info['min_samples']}"

    return text


def visualize_tracking_2d_dynamic_bin_cluster_text(tracked_objects, pcd_folder, cluster_params=None,
                                                   output_path='output_2d_cluster-detect5.mp4',
                                                   delay=0.001, interval=1, show=False):
    fig, ax = plt.subplots(figsize=(10, 10), facecolor='white')
    ax.set_facecolor('white')
    ax.set_aspect('equal')

    first_frame = True
    video_writer = None
    colors = [(1, 0, 0), (1, 1, 0), (0, 1, 0),
              (0, 1, 1), (0, 0, 1), (1, 0, 1)]

    pcd_file_paths = sorted(glob(os.path.join(pcd_folder, '*.bin')), key=natural_sort_key)
    print(f"Found {len(pcd_file_paths)} PCD files")

    try:
        for frame_id, objects in tqdm(enumerate(tracked_objects)):
            if frame_id % interval == 0:
                ax.clear()
                ax.set_facecolor('white')

                # 读取点云数据
                pcd_array = read_bin(pcd_file_paths[frame_id], 7)
                cluster_ids = pcd_array[:, 6].astype(int)
                source = pcd_array[:, 5]
                unique_clusters = np.unique(cluster_ids[cluster_ids != -1])

                # 绘制未分类点
                mask_noise = cluster_ids == -1
                ax.scatter(pcd_array[mask_noise, 0],
                           pcd_array[mask_noise, 1],
                           c='gray', s=0.5, alpha=0.3)

                # 收集所有cluster信息
                info_lines = []

                # 绘制各个cluster
                for i, cluster_id in enumerate(unique_clusters):
                    mask = cluster_ids == cluster_id
                    color = colors[i % len(colors)]

                    # 绘制点云
                    ax.scatter(pcd_array[mask, 0],
                               pcd_array[mask, 1],
                               c=[color], s=0.5, alpha=0.5)

                    # 获取cluster参数和信息
                    cluster_param = None
                    if cluster_params is not None and frame_id < len(cluster_params):
                        if i < len(cluster_params[frame_id]):
                            cluster_param = cluster_params[frame_id][i]

                    cluster_info = compute_cluster_info(
                        pcd_array[:, :3],
                        mask,
                        source,
                        cluster_param
                    )

                    # 显示cluster ID
                    center_x = np.mean(pcd_array[mask, 0])
                    center_y = np.mean(pcd_array[mask, 1])
                    ax.text(center_x, center_y + 2, f"{cluster_id}",
                            fontsize=10, color=color,
                            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

                    # 添加信息行
                    info_lines.append(format_bbox_info(cluster_id, cluster_info))

                # 在左上角显示所有信息（每个cluster一行）
                info_text = '\n'.join(info_lines)
                ax.text(0.02, 0.98, info_text,
                        transform=ax.transAxes,
                        verticalalignment='top',
                        fontsize=12,
                        color='black',
                        bbox=dict(facecolor='white',
                                  alpha=0.8,
                                  edgecolor='black',
                                  boxstyle='round,pad=0.5'))

                # 绘制跟踪框
                for obj in objects:
                    bbox = obj[:7]
                    rect = patches.Rectangle(
                        (bbox[0] - bbox[3] / 2, bbox[1] - bbox[4] / 2),
                        bbox[3], bbox[4],
                        angle=np.rad2deg(bbox[6]),
                        edgecolor='red',
                        facecolor='none',
                        alpha=1,
                        linewidth=1.0,
                        rotation_point='center'
                    )
                    ax.add_patch(rect)

                ax.set_xlim(-50, 50)
                ax.set_ylim(-50, 50)
                ax.grid(True, alpha=0.3)
                ax.set_title(f'Frame: {frame_id}', fontsize=10)

                plt.tight_layout()

                # 保存帧
                fig.canvas.draw()
                buf = fig.canvas.buffer_rgba()
                w, h = fig.canvas.get_width_height()
                frame = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4)
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)

                if first_frame:
                    height, width = frame_bgr.shape[:2]
                    video_writer = cv2.VideoWriter(
                        output_path,
                        cv2.VideoWriter_fourcc(*'mp4v'),
                        20, (width, height)
                    )
                    first_frame = False

                video_writer.write(frame_bgr)
                if show and delay > 0:
                    plt.pause(delay)

    finally:
        if video_writer is not None:
            video_writer.release()
        plt.close()

def visualize_tracking_2d_dynamic_bin_cluster(tracked_objects, pcd_folder, output_path='output_2d_cluster-detect2.mp4',
                                      delay=0.001, interval=1, show=False):
    """
    Visualize tracking results with point cloud background and cluster visualization
    Args:
        tracked_objects: tracking results
        pcd_folder: folder containing point cloud files
        frequency: frame rate
        output_path: output video path
    """
    fig, ax = plt.subplots(figsize=(8, 8), facecolor='white')
    ax.set_facecolor('white')
    ax.set_aspect('equal')

    first_frame = True
    video_writer = None

    # 为cluster准备颜色映射
    colors = [
        (1, 0, 0),  # 红色
        (1, 1, 0),  # 黄色
        (0, 1, 0),  # 绿色
        (0, 1, 1),  # 青色
        (0, 0, 1),  # 蓝色
        (1, 0, 1)  # 品红
    ]

    pcd_file_paths = sorted(glob(os.path.join(pcd_folder, '*.bin')), key=natural_sort_key)
    # print(pcd_file_paths)
    print(f"Found {len(pcd_file_paths)} PCD files")
    try:
        for frame_id, objects in tqdm(enumerate(tracked_objects)):
            if frame_id % interval == 0:
                ax.clear()
                ax.set_facecolor('white')

                # Plot point cloud background with cluster colors
                pcd_array = read_bin(pcd_file_paths[frame_id], 7)  # 假设第7列是cluster id
                # print(pcd_array[:10])
                cluster_ids = pcd_array[:, 6].astype(int)
                unique_clusters = np.unique(cluster_ids[cluster_ids != -1])

                # 先绘制未分类的点（cluster_id = -1）
                mask_noise = cluster_ids == -1
                ax.scatter(pcd_array[mask_noise, 0],
                           pcd_array[mask_noise, 1],
                           c='gray',
                           s=0.5,
                           alpha=0.3)

                # 绘制各个cluster的点
                for i, cluster_id in enumerate(unique_clusters):
                    mask = cluster_ids == cluster_id
                    color = colors[i % len(colors)]
                    ax.scatter(pcd_array[mask, 0],
                               pcd_array[mask, 1],
                               c=[color],
                               s=0.5,
                               alpha=0.5)

                for obj in objects:
                    bbox = obj[:7]
                    box_color = 'red'

                    rect = patches.Rectangle(
                        (bbox[0] - bbox[3] / 2, bbox[1] - bbox[4] / 2),
                        bbox[3], bbox[4],
                        angle=np.rad2deg(bbox[6]),
                        edgecolor=box_color,
                        facecolor='none',
                        alpha=1,
                        linewidth=1.0,
                        rotation_point='center'
                    )
                    ax.add_patch(rect)

                ax.set_xlim(-50, 50)
                ax.set_ylim(-50, 50)
                ax.grid(True, alpha=0.3)
                ax.set_title(f'Frame: {frame_id}', fontsize=10)

                plt.tight_layout()
                fig.canvas.draw()
                buf = fig.canvas.buffer_rgba()
                w, h = fig.canvas.get_width_height()
                frame = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4)
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)

                if first_frame:
                    height, width = frame_bgr.shape[:2]
                    video_writer = cv2.VideoWriter(
                        output_path,
                        cv2.VideoWriter_fourcc(*'mp4v'),
                        20,
                        (width, height)
                    )
                    first_frame = False

                video_writer.write(frame_bgr)
                if show and delay > 0:
                    plt.pause(delay)
    finally:
        if video_writer is not None:
            video_writer.release()
        plt.close()

def visualize_tracking_2d_dynamic_bin_dynamic(tracked_objects, pcd_folder, output_path='output_2d_dynamic-detect.mp4',
                                      delay=0.001, interval=1, show=False):
    """
    Visualize tracking results with point cloud background and cluster visualization
    Args:
        tracked_objects: tracking results
        pcd_folder: folder containing point cloud files
        frequency: frame rate
        output_path: output video path
    """
    fig, ax = plt.subplots(figsize=(8, 8), facecolor='white')
    ax.set_facecolor('white')
    ax.set_aspect('equal')

    first_frame = True
    video_writer = None

    pcd_file_paths = sorted(glob(os.path.join(pcd_folder, '*.bin')), key=natural_sort_key)
    # print(pcd_file_paths)
    print(f"Found {len(pcd_file_paths)} PCD files")
    try:
        for frame_id, objects in tqdm(enumerate(tracked_objects)):
            if frame_id % interval == 0:
                ax.clear()
                ax.set_facecolor('white')

                # Plot point cloud background with cluster colors
                pcd_array = read_bin(pcd_file_paths[frame_id], 7)  # 假设第7列是cluster id
                # print(pcd_array[:10])
                is_dynamic = pcd_array[:, 4].astype(int) # 假设第4列是dynamic标志

                mask_static = is_dynamic == 0
                mask_dynamic = is_dynamic == 1
                ax.scatter(pcd_array[mask_static, 0],
                           pcd_array[mask_static, 1],
                           c='gray',
                           s=0.5,
                           alpha=0.3)

                ax.scatter(pcd_array[mask_dynamic, 0],
                           pcd_array[mask_dynamic, 1],
                           c='red',
                           s=0.5,
                           alpha=0.5)

                ax.set_xlim(-50, 50)
                ax.set_ylim(-50, 50)
                ax.grid(True, alpha=0.3)
                ax.set_title(f'Frame: {frame_id}', fontsize=10)

                plt.tight_layout()
                fig.canvas.draw()
                buf = fig.canvas.buffer_rgba()
                w, h = fig.canvas.get_width_height()
                frame = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4)
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)

                if first_frame:
                    height, width = frame_bgr.shape[:2]
                    video_writer = cv2.VideoWriter(
                        output_path,
                        cv2.VideoWriter_fourcc(*'mp4v'),
                        20,
                        (width, height)
                    )
                    first_frame = False

                video_writer.write(frame_bgr)
                if show and delay > 0:
                    plt.pause(delay)
    finally:
        if video_writer is not None:
            video_writer.release()
        plt.close()

def visualize_tracking_2d(tracked_objects, pcd_folder, frequency, output_path='tracking.mp4',
                                  km_per_h=True, arrow=False, delay=0.001, interval=1, show=False,confidence_threshold=0.9,
                                      columns=['x', 'y', 'z', 'range', 'reflectivity', 'near_ir']):
    """
    Visualize tracking results with point cloud background
    Args:
        tracked_objects: tracking results
        pc_data: list of point clouds for each frame [(N, 2), ...] containing x,y coordinates
        frequency: frame rate
        output_path: output video path
    """
    fig, ax = plt.subplots(figsize=(8, 8), facecolor='white')
    ax.set_facecolor('white')
    ax.set_aspect('equal')

    first_frame = True
    video_writer = None
    trajectories = {}
    labels = ['person', 'small vehicle', 'large vehicle', 'other']
    if km_per_h:
        frequency = frequency * 3.6
    pcd_file_paths = sorted(glob(os.path.join(pcd_folder, '*.bin')))
    print(f"Found {len(pcd_file_paths)} PCD files")
    try:
        for frame_id,objects in tqdm(enumerate(tracked_objects)):
            if frame_id % interval == 0:
                ax.clear()
                ax.set_facecolor('white')

                # Plot point cloud background
                pcd_array = read_bin(pcd_file_paths[frame_id], len(columns))
                ax.scatter(pcd_array[:, 0], pcd_array[:, 1], c='gray', s=0.5, alpha=0.3)

                active_ids = []
                for obj in objects:
                    bbox = obj['bbox']
                    track_id = obj['track_id']
                    speed = obj.get('speed', 0)
                    label = obj.get('label', 0)
                    confidence = obj.get('confidence_score', 1.0)

                    speed_magnitude = np.linalg.norm(speed) * frequency if isinstance(speed,
                                                                                      np.ndarray) else speed * frequency
                    active_ids.append(track_id)

                    # 根据confidence score选择颜色
                    box_color = 'red' if confidence >= confidence_threshold else 'orange'

                    rect = patches.Rectangle(
                        (bbox[0] - bbox[3] / 2, bbox[1] - bbox[4] / 2),
                        bbox[3], bbox[4],
                        angle=np.rad2deg(bbox[6]),
                        edgecolor=box_color,
                        facecolor='none',
                        alpha=1,
                        linewidth=1.0,
                        rotation_point='center'
                    )
                    ax.add_patch(rect)

                    if arrow:
                        arrow_length = 3.0
                        dx = arrow_length * np.cos(bbox[6])
                        dy = arrow_length * np.sin(bbox[6])
                        ax.arrow(bbox[0], bbox[1], dx, dy,
                                 head_width=0.3, head_length=0.6,
                                 fc=box_color, ec=box_color)

                    # 添加confidence score到文本标注
                    ax.text(bbox[0] + bbox[3] / 2, bbox[1] + bbox[4] / 2,
                            f'{track_id}\n{speed_magnitude:.1f}\n{labels[label]}\nconf:{confidence:.2f}',
                            color=box_color, fontsize=8,
                            horizontalalignment='center',
                            verticalalignment='bottom')

                    if track_id not in trajectories:
                        trajectories[track_id] = []
                    trajectories[track_id].append([bbox[0], bbox[1]])
                    if track_id in trajectories:
                        trajectory = np.array(trajectories[track_id])
                        ax.plot(trajectory[:, 0], trajectory[:, 1],
                                '-', color=box_color, linewidth=0.8, alpha=0.3)

                trajectories = {k: v for k, v in trajectories.items() if k in active_ids}

                ax.set_xlim(-50, 50)
                ax.set_ylim(-50, 50)
                ax.grid(True, alpha=0.3)
                ax.set_title(f'Frame: {frame_id}', fontsize=10)

                plt.tight_layout()
                fig.canvas.draw()
                buf = fig.canvas.buffer_rgba()
                w, h = fig.canvas.get_width_height()
                frame = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4)
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)

                if first_frame:
                    height, width = frame_bgr.shape[:2]
                    video_writer = cv2.VideoWriter(
                        output_path,
                        cv2.VideoWriter_fourcc(*'mp4v'),
                        20,
                        (width, height)
                    )
                    first_frame = False

                video_writer.write(frame_bgr)
                if show and delay > 0:
                    plt.pause(delay)
    finally:
        if video_writer is not None:
            video_writer.release()
        plt.close()

def visualize_tracking_3d_dynamic_bin(tracked_objects, pcd_folder,
                                      output_path='output_3d.mp4', z_min=-0.1, z_max=0.1,
                                      ground=False, delay=0.001, interval=1, show_trajectory=True,
                                      confidence_threshold=0.7):
    vis = o3d.visualization.Visualizer()
    vis.create_window(height=800, width=800)
    view_control = vis.get_view_control()
    opt = vis.get_render_option()
    opt.background_color = np.array([1, 1, 1])
    opt.point_size = 1.0

    view_control.set_zoom(0.01)
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
    vis.add_geometry(coordinate_frame)

    pcd_file_paths = sorted(glob(os.path.join(pcd_folder, '*.bin')))
    print(f"Found {len(pcd_file_paths)} PCD files")

    trajectories = {}
    trajectory_lines_dict = {}
    first_frame = True
    video_writer = None

    try:
        for frame_id, objects in tqdm(enumerate(tracked_objects[:])):
            if frame_id % interval == 0:
                vis.clear_geometries()
                vis.add_geometry(coordinate_frame)

                # 处理点云
                pcd_array = read_bin(pcd_file_paths[frame_id], 3)
                pcd = create_pcd(pcd_array)
                points = np.asarray(pcd.points)
                colors = np.full_like(points, [0.5, 0.5, 0.5])
                if ground:
                    ground_points_mask = (points[:, 2] >= z_min) & (points[:, 2] <= z_max)
                    colors[ground_points_mask] = [0, 0, 1]
                pcd.colors = o3d.utility.Vector3dVector(colors)
                vis.add_geometry(pcd)

                active_ids = []
                for obj in objects:
                    bbox = obj['bbox']
                    track_id = obj['track_id']
                    confidence = obj.get('confidence_score', 1.0)
                    active_ids.append(track_id)

                    # 根据confidence score设置颜色
                    box_color = (1, 0, 0) if confidence >= confidence_threshold else (1, 0.5, 0)  # red or orange

                    # 添加bbox
                    center = np.array([bbox[0], bbox[1], bbox[2]])
                    extent = np.array([bbox[3], bbox[4], bbox[5]])
                    R = o3d.geometry.get_rotation_matrix_from_xyz((0, 0, bbox[6]))
                    obb = o3d.geometry.OrientedBoundingBox(center, R, extent)
                    obb.color = box_color
                    vis.add_geometry(obb)

                    # 更新轨迹
                    if track_id not in trajectories:
                        trajectories[track_id] = []
                        trajectory_lines_dict[track_id] = o3d.geometry.LineSet()
                    trajectories[track_id].append([bbox[0], bbox[1], bbox[2]])

                    if show_trajectory and len(trajectories[track_id]) > 1:
                        points = np.array(trajectories[track_id])
                        lines = [[i, i + 1] for i in range(len(points) - 1)]
                        line_colors = [[box_color[0], box_color[1], box_color[2]] for _ in lines]
                        current_lines = trajectory_lines_dict[track_id]
                        current_lines.points = o3d.utility.Vector3dVector(points)
                        current_lines.lines = o3d.utility.Vector2iVector(lines)
                        current_lines.colors = o3d.utility.Vector3dVector(line_colors)
                        vis.add_geometry(current_lines)

                trajectories = {k: v for k, v in trajectories.items() if k in active_ids}
                trajectory_lines_dict = {k: v for k, v in trajectory_lines_dict.items() if k in active_ids}

                view_control.set_zoom(0.2)
                vis.poll_events()
                vis.update_renderer()

                img = vis.capture_screen_float_buffer()
                img = np.asarray(img)
                img = (img * 255).astype(np.uint8)
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                if first_frame:
                    height, width = img_bgr.shape[:2]
                    video_writer = cv2.VideoWriter(
                        output_path,
                        cv2.VideoWriter_fourcc(*'mp4v'),
                        20,
                        (width, height)
                    )
                    first_frame = False

                video_writer.write(img_bgr)
                if delay > 0:
                    time.sleep(delay)
                if not vis.poll_events():
                    break
    finally:
        if video_writer is not None:
            video_writer.release()
        vis.destroy_window()

def visualize_tracking_3d_static(tracked_objects, pcd_folder,transform_matrix_path,z_min=-0.1, z_max=0.1, ground=False,interval=1):
    """
    Visualize the tracking results in 3D.
    :param bbox_dict:
    :param frame_id_mapping:
    :param pcd_folder:
    :param transform_matrix:
    :param z_min:
    :param z_max:
    :param ground:
    :param delay:
    :return:
    """
    # Get all PCD files
    pcd_file_paths = sorted(glob(os.path.join(pcd_folder, '*.pcd')))
    print(f"Found {len(pcd_file_paths)} PCD files")
    # Load the transformation matrix
    transform_matrix = np.load(transform_matrix_path)
    for frame_id, objects in enumerate(tracked_objects):
        if frame_id % interval == 0:
            # 创建可视化器
            vis = o3d.visualization.Visualizer()
            vis.create_window(height=800, width=800)
            # 设置默认视角
            view_control = vis.get_view_control()
            # 设置渲染选项
            opt = vis.get_render_option()
            opt.background_color = np.array([1, 1, 1])  # 白色背景
            opt.point_size = 1.0
            # 添加坐标系
            coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
            vis.add_geometry(coordinate_frame)
            # 读取并处理点云
            pcd = read_pcd_custom(pcd_file_paths[frame_id])
            # 应用变换矩阵
            pcd.transform(transform_matrix)
            # 设置点云颜色
            points = np.asarray(pcd.points)
            colors = np.full_like(points, [0.5, 0.5, 0.5])  # 灰色点云
            if ground:
                ground_points_mask = (points[:, 2] >= z_min) & (points[:, 2] <= z_max)
                colors[ground_points_mask] = [0, 0, 1]  # 蓝色地面点
            pcd.colors = o3d.utility.Vector3dVector(colors)
            # 添加点云
            vis.add_geometry(pcd)
            # 添加当前帧的所有bbox
            for obj in objects:
                bbox = obj['bbox']  # [x, y, z, w, l, h, yaw]
                center = np.array([bbox[0], bbox[1], bbox[2]])
                extent = np.array([bbox[3], bbox[4], bbox[5]])
                R = o3d.geometry.get_rotation_matrix_from_xyz((0, 0, bbox[6]))
                obb = o3d.geometry.OrientedBoundingBox(center, R, extent)
                # print(obb)
                obb.color = (1, 0, 0)  # 红色bbox
                vis.add_geometry(obb)
            # 正确的俯视角度
            view_control.set_zoom(0.2)  # 调整缩放比例
            # 运行可视化器
            vis.run()
            # 关闭可视化器
            vis.destroy_window()


def fuse_visualize(pcd_source_array, pcd_target_array):
    """
    Visualize the fused point cloud with coordinate frame, grid, and camera control.

    Args:
        pcd_source_array: Source point cloud array
        pcd_target_array: Target point cloud array
        zoom: Zoom level (default: 0.7, smaller value means zoom out)
        front_view: Camera front direction (default: None, will use [0, 0, -1])
        up_vector: Camera up direction (default: None, will use [0, 1, 0])
    """
    # 使用不同颜色显示源点云和目标点云
    pcd_source = create_pcd(pcd_source_array)
    pcd_target = create_pcd(pcd_target_array)
    pcd_source.paint_uniform_color([1, 0, 0])  # 红色为源点云
    pcd_target.paint_uniform_color([0, 0, 1])  # 蓝色为目标点云
    # 创建新的可视化窗口
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    # 设置渲染选项
    opt = vis.get_render_option()
    opt.background_color = np.array([1, 1, 1])  # 白色背景
    opt.point_size = 1.5  # 点的大小
    # 添加点云
    vis.add_geometry(pcd_source)
    vis.add_geometry(pcd_target)
    # 创建并添加目标点云坐标系 - 在原点(0,0,0)
    target_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=5, origin=[0, 0, 0])  # Red:x, Green:y, Blue:z
    vis.add_geometry(target_frame)
    # 创建并添加网格 - 网格间距为5单位
    grid_size = 5  # 每个网格单元的大小
    grid_extent = 50  # 网格在每个方向的范围
    grid_color = [0.5, 0.5, 0.5]  # 灰色
    # 创建XY平面网格（地面网格）
    for i in range(-grid_extent, grid_extent + 1, grid_size):
        # 创建X方向线
        line_x = o3d.geometry.LineSet()
        points_x = np.array([
            [i, -grid_extent, 0],
            [i, grid_extent, 0]
        ])
        lines_x = np.array([[0, 1]])
        line_x.points = o3d.utility.Vector3dVector(points_x)
        line_x.lines = o3d.utility.Vector2iVector(lines_x)
        line_x.colors = o3d.utility.Vector3dVector([grid_color])
        vis.add_geometry(line_x)
        # 创建Y方向线
        line_y = o3d.geometry.LineSet()
        points_y = np.array([
            [-grid_extent, i, 0],
            [grid_extent, i, 0]
        ])
        lines_y = np.array([[0, 1]])
        line_y.points = o3d.utility.Vector3dVector(points_y)
        line_y.lines = o3d.utility.Vector2iVector(lines_y)
        line_y.colors = o3d.utility.Vector3dVector([grid_color])
        vis.add_geometry(line_y)
    # 设置视角和缩放
    view_control = vis.get_view_control()
    # 设置视角方向
    front_view = [0, 0, 1]
    up_vector = [0, 1, 0]
    zoom = 0.2 # Smaller value means zoom in
    # 需要先运行一次update以便视角设置生效
    vis.poll_events()
    vis.update_renderer()
    # 设置相机参数
    view_control.set_front(front_view)
    view_control.set_up(up_vector)
    view_control.set_zoom(zoom)
    # 运行可视化器
    vis.run()
    # 关闭可视化器
    vis.destroy_window()

def fuse_visualize_video(pcd_folder, output_path='30.mp4',
                                      delay=0.001, interval=1, show=False):
    """
    Visualize tracking results with point cloud background and cluster visualization
    Args:
        tracked_objects: tracking results
        pcd_folder: folder containing point cloud files
        frequency: frame rate
        output_path: output video path
    """
    fig, ax = plt.subplots(figsize=(8, 8), facecolor='white')
    ax.set_facecolor('white')
    ax.set_aspect('equal')

    first_frame = True
    video_writer = None

    pcd_file_paths = sorted(glob(os.path.join(pcd_folder, '*.bin')), key=natural_sort_key)
    # print(pcd_file_paths)
    print(f"Found {len(pcd_file_paths)} PCD files")
    try:
        for frame_id, objects in tqdm(enumerate(tracked_objects)):
            if frame_id % interval == 0:
                ax.clear()
                ax.set_facecolor('white')

                # Plot point cloud background with cluster colors
                pcd_array = read_bin(pcd_file_paths[frame_id], 3)  # 假设第7列是cluster id
                # print(pcd_array[:10])
                ax.scatter(pcd_array[mask_static, 0],
                           pcd_array[mask_static, 1],
                           c='gray',
                           s=0.5,
                           alpha=0.3)

                ax.scatter(pcd_array[mask_dynamic, 0],
                           pcd_array[mask_dynamic, 1],
                           c='red',
                           s=0.5,
                           alpha=0.5)

                ax.set_xlim(-50, 50)
                ax.set_ylim(-50, 50)
                ax.grid(True, alpha=0.3)
                ax.set_title(f'Frame: {frame_id}', fontsize=10)

                plt.tight_layout()
                fig.canvas.draw()
                buf = fig.canvas.buffer_rgba()
                w, h = fig.canvas.get_width_height()
                frame = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4)
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)

                if first_frame:
                    height, width = frame_bgr.shape[:2]
                    video_writer = cv2.VideoWriter(
                        output_path,
                        cv2.VideoWriter_fourcc(*'mp4v'),
                        20,
                        (width, height)
                    )
                    first_frame = False

                video_writer.write(frame_bgr)
                if show and delay > 0:
                    plt.pause(delay)
    finally:
        if video_writer is not None:
            video_writer.release()
        plt.close()

def visualize_frame(pcd_array, cluster_ids, bboxes, frame_id, classes):
    """单帧可视化函数"""
    fig, ax = plt.subplots(figsize=(10, 10), facecolor='white')
    ax.set_facecolor('white')
    ax.set_aspect('equal')
    colors = [(1, 0, 0), (1, 1, 0), (0, 1, 0),
              (0, 1, 1), (0, 0, 1), (1, 0, 1)]
    # 绘制未分类点
    mask_noise = cluster_ids == -1
    ax.scatter(pcd_array[mask_noise, 0],
               pcd_array[mask_noise, 1],
               c='gray', s=0.5, alpha=0.3)
    # 绘制各个cluster
    unique_clusters = np.unique(cluster_ids[cluster_ids != -1])
    for i, cluster_id in enumerate(unique_clusters):
        mask = cluster_ids == cluster_id
        color = colors[i % len(colors)]
        # 绘制点云
        ax.scatter(pcd_array[mask, 0],pcd_array[mask, 1], c=[color], s=0.5, alpha=0.5)
    # 绘制边界框
    for bbox in bboxes:
        rect = patches.Rectangle(
            (bbox['x'] - bbox['w'] / 2, bbox['y'] - bbox['l'] / 2),
            bbox['w'], bbox['l'],
            angle=np.rad2deg(bbox['yaw']),
            edgecolor='red',
            facecolor='none',
            alpha=1,
            linewidth=1.0,
            rotation_point='center'
        )
        ax.add_patch(rect)
        text = f"{classes[bbox['label']]},{bbox['confidence']:.6f}"
        ax.text(bbox['x'] + bbox['w'] / 2, bbox['y'] + bbox['l'] / 2,
                text, color='black', fontsize=10,
                horizontalalignment='center',
                verticalalignment='bottom')
    ax.set_xlim(-60, 60)
    ax.set_ylim(-60, 60)
    ax.grid(True, alpha=0.3)
    ax.set_title(f'Frame: {frame_id}', fontsize=10)
    plt.tight_layout()
    plt.show()

def visualize_frame_for_video(pcd_array, cluster_ids, bboxes, frame_id, classes, fig, ax):
    """单帧可视化函数，针对视频创建进行优化"""
    ax.clear()
    ax.set_facecolor('white')
    ax.set_aspect('equal')
    colors = [(1, 0, 0), (1, 1, 0), (0, 1, 0),
              (0, 1, 1), (0, 0, 1), (1, 0, 1)]

    # 绘制未分类点
    mask_noise = cluster_ids == -1
    ax.scatter(pcd_array[mask_noise, 0],
               pcd_array[mask_noise, 1],
               c='gray', s=0.5, alpha=0.3)

    # 绘制各个cluster
    unique_clusters = np.unique(cluster_ids[cluster_ids != -1])
    for i, cluster_id in enumerate(unique_clusters):
        mask = cluster_ids == cluster_id
        color = colors[i % len(colors)]
        # 绘制点云
        ax.scatter(pcd_array[mask, 0], pcd_array[mask, 1], c=[color], s=0.5, alpha=0.5)

    # 绘制边界框
    for bbox in bboxes:
        rect = patches.Rectangle(
            (bbox['x'] - bbox['w'] / 2, bbox['y'] - bbox['l'] / 2),
            bbox['w'], bbox['l'],
            angle=np.rad2deg(bbox['yaw']),
            edgecolor='red',
            facecolor='none',
            alpha=1,
            linewidth=1.0,
            rotation_point='center'
        )
        ax.add_patch(rect)
        text = f"{classes[bbox['label']]},{bbox['confidence']:.2f}"
        ax.text(bbox['x'] + bbox['w'] / 2, bbox['y'] + bbox['l'] / 2,
                text, color='black', fontsize=10,
                horizontalalignment='center',
                verticalalignment='bottom')

    ax.set_xlim(-50, 50)
    ax.set_ylim(-50, 50)
    ax.grid(True, alpha=0.3)
    ax.set_title(f'Frame: {frame_id}', fontsize=10)
    return fig


def create_point_cloud_video(pcd_files, output_video_path, process_frame_func, args, fps=10):
    """
    创建点云视频函数

    参数:
    - pcd_files: 点云文件路径列表
    - output_video_path: 输出视频文件路径
    - process_frame_func: 处理单帧的函数（返回bboxes和处理后的点云）
    - args: 传递给process_frame_func的参数
    - fps: 视频帧率，默认10
    """
    print("=== Creating Point Cloud Video ===")

    # 确保输出目录存在
    output_dir = os.path.dirname(output_video_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 创建图像对象（只创建一次，重复使用以提高性能）
    fig, ax = plt.subplots(figsize=(10, 10), facecolor='white')
    plt.tight_layout()
    canvas = FigureCanvas(fig)

    # 设置视频编码器和输出
    temp_frames_dir = os.path.join(output_dir, 'temp_frames')
    if not os.path.exists(temp_frames_dir):
        os.makedirs(temp_frames_dir)

    columns_len = len(args.columns)
    classes = ['person', 'small vehicle', 'large vehicle', 'other']

    print(f"Processing {len(pcd_files)} frames...")
    for i, pcd_file_path in tqdm(enumerate(pcd_files)):
        frame_name = os.path.basename(pcd_file_path)

        # 读取并处理点云
        pcd_array = read_bin(pcd_file_path, columns_len)

        # 过滤点云数据
        filter_mask = (pcd_array[:, args.columns.index('z')] >= 0.1) & (
                pcd_array[:, args.columns.index('z')] <= 3) & (
                              pcd_array[:, args.columns.index('dynamic_mask')] == 1)
        filtered_pcd_array = pcd_array[filter_mask]

        # 处理单帧
        frame_bboxes, processed_pcd_array = process_frame_func(i, frame_name, pcd_array, filtered_pcd_array, args)

        # 可视化当前帧
        cluster_id = processed_pcd_array[:, -1]  # 假设聚类ID被添加为最后一列
        fig = visualize_frame_for_video(processed_pcd_array, cluster_id, frame_bboxes, i, classes, fig, ax)

        # 将图转换为图像
        canvas.draw()
        image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        # 保存帧
        frame_path = os.path.join(temp_frames_dir, f'frame_{i:06d}.jpg')
        cv2.imwrite(frame_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    # 使用OpenCV创建视频
    print("Creating video...")
    frame_sample = cv2.imread(os.path.join(temp_frames_dir, f'frame_{0:06d}.jpg'))
    height, width, _ = frame_sample.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 编码器
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # 将所有帧添加到视频
    for i in tqdm(range(len(pcd_files))):
        frame_path = os.path.join(temp_frames_dir, f'frame_{i:06d}.jpg')
        frame = cv2.imread(frame_path)
        video_writer.write(frame)

    # 释放资源
    video_writer.release()
    plt.close(fig)

    # 可选：清理临时帧文件
    import shutil
    shutil.rmtree(temp_frames_dir)

    print(f"Video created successfully: {output_video_path}")
    return output_video_path

def visualize_frame_3d(pcd_array, cluster_ids, bboxes, frame_id):
    """
    使用Open3D进行3D点云和边界框可视化

    参数:
        pcd_array: 包含点云数据的数组，形状为(N, 3+)，前三列是x,y,z坐标
        cluster_ids: 每个点的聚类ID，形状为(N,)，未分类点标记为-1
        bboxes: 边界框列表，每个边界框是一个字典或包含[x,y,z,w,l,h,yaw]的列表
        frame_id: 当前帧ID，用于显示标题
        save_image: 是否保存图像，默认为False
        output_path: 图像保存路径，默认为None
    """
    # 创建可视化窗口
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=f"Frame {frame_id}", width=1280, height=720)
    # 创建点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcd_array[:, :3])
    # 设置点云颜色 - 首先为未分类点设置灰色
    colors = np.ones((len(pcd_array), 3)) * 0.5  # 默认灰色
    # 为每个聚类设置不同颜色
    unique_clusters = np.unique(cluster_ids[cluster_ids != -1])
    color_map = plt.cm.get_cmap('tab10', max(10, len(unique_clusters)))
    for i, cluster_id in enumerate(unique_clusters):
        mask = cluster_ids == cluster_id
        color = color_map(i % 10)[:3]  # 获取RGB颜色，去掉alpha通道
        colors[mask] = color
    pcd.colors = o3d.utility.Vector3dVector(colors)
    # 添加点云到可视化窗口
    vis.add_geometry(pcd)
    # 创建和添加边界框
    for i, bbox in enumerate(bboxes):
        # 处理不同的边界框格式
        if isinstance(bbox, dict):
            center = [bbox['x'], bbox['y'], bbox['z']]
            size = [bbox['w'], bbox['l'], bbox['h']]
            yaw = bbox['yaw']
        else:  # 假设是列表 [x,y,z,w,l,h,yaw]
            center = [bbox[0], bbox[1], bbox[2]]
            size = [bbox[3], bbox[4], bbox[5]]
            yaw = bbox[6]
        # 创建Open3D边界框对象
        bbox_o3d = create_oriented_bounding_box(center, size, yaw)
        # 为每个边界框设置不同的颜色
        color_idx = i % len(unique_clusters) if len(unique_clusters) > 0 else 0
        bbox_color = color_map(color_idx)[:3]
        # 注意：LineSet的colors是为每条线设置颜色，而不是为顶点
        bbox_o3d.colors = o3d.utility.Vector3dVector(
            [[bbox_color[0], bbox_color[1], bbox_color[2]] for _ in range(len(bbox_o3d.lines))])
        # 添加边界框到可视化窗口
        vis.add_geometry(bbox_o3d)
    # 设置视角
    ctr = vis.get_view_control()
    ctr.set_zoom(0.1)  # 调整缩放比例,数字越小，视角越近
    ctr.set_front([0, 0, 1])  # 正视图
    ctr.set_lookat([0, 0, 0])  # 看向原点
    ctr.set_up([0, 1, 0])  # 上方向
    # 设置渲染选项
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0.9, 0.9, 0.9])  # 浅灰色背景
    opt.point_size = 3
    # 更新可视化
    vis.poll_events()
    vis.update_renderer()
    # 显示窗口并等待关闭
    vis.run()
    vis.destroy_window()

def create_oriented_bounding_box(center, size, yaw):
    """
    创建带方向的边界框

    参数:
        center: 边界框中心坐标 [x, y, z]
        size: 边界框尺寸 [width, length, height]
        yaw: 绕z轴的旋转角度（弧度）

    返回:
        o3d.geometry.LineSet: Open3D线集对象表示的边界框
    """
    # 创建边界框的8个顶点坐标（相对于中心点）
    w, l, h = size
    w_half, l_half, h_half = w / 2, l / 2, h / 2
    # 未旋转的边界框顶点（相对于中心）
    vertices = [
        [-w_half, -l_half, -h_half],  # 0: 左下后
        [w_half, -l_half, -h_half],  # 1: 右下后
        [w_half, l_half, -h_half],  # 2: 右上后
        [-w_half, l_half, -h_half],  # 3: 左上后
        [-w_half, -l_half, h_half],  # 4: 左下前
        [w_half, -l_half, h_half],  # 5: 右下前
        [w_half, l_half, h_half],  # 6: 右上前
        [-w_half, l_half, h_half]  # 7: 左上前
    ]
    # 边界框的边（通过顶点索引定义）
    lines = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # 底面
        [4, 5], [5, 6], [6, 7], [7, 4],  # 顶面
        [0, 4], [1, 5], [2, 6], [3, 7]  # 侧面的连接线
    ]
    # 创建旋转矩阵
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([
        [c, -s, 0],
        [s, c, 0],
        [0, 0, 1]
    ])
    # 应用旋转并平移到中心位置
    rotated_vertices = []
    for vertex in vertices:
        rotated = np.dot(R, vertex)
        translated = [rotated[0] + center[0], rotated[1] + center[1], rotated[2] + center[2]]
        rotated_vertices.append(translated)
    # 创建Open3D线集
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(rotated_vertices)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    # 设置线的颜色 - 默认红色
    line_colors = [[1, 0, 0] for _ in range(len(lines))]
    line_set.colors = o3d.utility.Vector3dVector(line_colors)
    return line_set

def oriented_bounding_box(center, size, yaw):
    """
    手动创建有向边界框，兼容所有版本的Open3D。
    Args:
        center: 边界框中心点 [x, y, z]
        size: 边界框尺寸 [width (x轴), length (y轴), height (z轴)]
        yaw: 绕Z轴旋转角（弧度）
    Returns:
        o3d.geometry.OrientedBoundingBox 对象
    """
    # 确保所有输入都是浮点数
    center = np.array(center, dtype=np.float64)
    size = np.array(size, dtype=np.float64)
    yaw = float(yaw)

    # 定义旋转矩阵
    R = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ], dtype=np.float64)

    # 定义 bbox corners（局部坐标系下）
    w, l, h = size[0] / 2, size[1] / 2, size[2] / 2
    corners = np.array([
        [-w, -l, -h],
        [w, -l, -h],
        [w, l, -h],
        [-w, l, -h],
        [-w, -l, h],
        [w, -l, h],
        [w, l, h],
        [-w, l, h]
    ], dtype=np.float64)

    # 旋转和平移corners
    corners = (R @ corners.T).T + center

    # 创建bbox - 使用线框显示而不是面
    # 先尝试使用create_from_points
    obb = o3d.geometry.OrientedBoundingBox.create_from_points(o3d.utility.Vector3dVector(corners))

    # 如果边界框无效，尝试直接设置参数
    if obb.volume() <= 0:
        try:
            # 尝试直接设置参数
            obb = o3d.geometry.OrientedBoundingBox(center=center, R=R, extent=size)
        except:
            # 创建线框来代替
            lines = o3d.geometry.LineSet()
            lines.points = o3d.utility.Vector3dVector(corners)
            # 定义线段连接
            lines.lines = o3d.utility.Vector2iVector([
                [0, 1], [1, 2], [2, 3], [3, 0],  # 底部矩形
                [4, 5], [5, 6], [6, 7], [7, 4],  # 顶部矩形
                [0, 4], [1, 5], [2, 6], [3, 7]  # 连接顶部和底部
            ])
            return lines

    return obb

def visualize_3d_bboxes(gt_data, det_data, pcd_folder, column_len):
    """
    3D可视化函数 - 在同一帧中显示真实值和检测结果的3D边界框
    真实值边界框：红色
    检测结果边界框：蓝色
    Args:
        gt_data: 真实值数据字典
        det_data: 检测结果数据字典
        pcd_folder: 点云文件所在文件夹
    """
    pcd_files = sorted(glob(os.path.join(pcd_folder, '*.bin')))
    for frame_id, pcd_file in enumerate(pcd_files):
        # print(f"处理帧 {frame_id + 1}/{len(pcd_files)}: {pcd_file}")
        frame_name = os.path.basename(pcd_file)
        print(frame_name)
        # 检查数据是否存在
        gt_bboxes = gt_data.get(frame_name, [])
        det_bboxes = det_data.get(frame_name, [])
        # print(f"真实框数量: {len(gt_bboxes)}, 检测框数量: {len(det_bboxes)}")
        if not gt_bboxes and not det_bboxes:
            continue
        # 读取点云数据
        pcd_array = read_bin(pcd_file, column_len)
        # 初始化可视化
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=f"Frame {frame_name}", width=1280, height=720)
        # 创建点云对象
        pcd = create_pcd(pcd_array)
        colors = np.ones((len(pcd_array), 3)) * 0.5
        pcd.colors = o3d.utility.Vector3dVector(colors)
        vis.add_geometry(pcd)
        # 收集所有有效中心点以便设置视角
        all_centers = []
        # print('添加真实框')
        # 添加真实值边界框（红色）和边界框线框（更好的可视化）
        for i, bbox in enumerate(gt_bboxes):
            center, size, yaw = parse_bbox(bbox)
            all_centers.append(center)
            # 创建线框
            lines = create_bbox_lineset(center, size, yaw, [1, 0, 0])  # 红色
            vis.add_geometry(lines)
            # 也创建标准边界框
            bbox_o3d = oriented_bounding_box(center, size, yaw)
            # print(f"真实框 {i}:", bbox_o3d)
            bbox_o3d.color = [1, 0, 0]  # 红色
            vis.add_geometry(bbox_o3d)
        # print('添加检测框')
        # 添加检测结果边界框（蓝色）
        for i, bbox in enumerate(det_bboxes):
            center, size, yaw = parse_bbox(bbox)
            all_centers.append(center)
            # 创建线框
            lines = create_bbox_lineset(center, size, yaw, [0, 0, 1])  # 蓝色
            vis.add_geometry(lines)
            # 也创建标准边界框
            bbox_o3d = oriented_bounding_box(center, size, yaw)
            # print(f"检测框 {i}:", bbox_o3d)
            bbox_o3d.color = [0, 0, 1]  # 蓝色
            vis.add_geometry(bbox_o3d)
        # 智能设置视角（确保看得到所有框）
        if all_centers:
            # 计算所有边界框的平均中心位置
            center_array = np.array(all_centers)
            center_mean = np.mean(center_array, axis=0)
            # 计算边界框到中心的最大距离
            max_dist = 0
            for center in all_centers:
                dist = np.linalg.norm(np.array(center) - center_mean)
                max_dist = max(max_dist, dist)
            # 设置视角
            ctr = vis.get_view_control()
            # 增大视图比例确保看到所有内容
            ctr.set_zoom(0.3)
            ctr.set_front([0, 0, 1])  # 从上方俯视
            ctr.set_lookat(center_mean)  # 看向所有边界框的平均位置
            ctr.set_up([0, 1, 0])
        else:
            # 默认视角
            ctr = vis.get_view_control()
            ctr.set_zoom(0.1)
            ctr.set_front([0, 0, 1])
            ctr.set_lookat([0, 0, 0])
            ctr.set_up([0, 1, 0])
        # 设置渲染选项
        opt = vis.get_render_option()
        opt.background_color = np.asarray([0.9, 0.9, 0.9])
        opt.point_size = 1
        opt.line_width = 5.0  # 增大线宽提高可见性
        vis.poll_events()
        vis.update_renderer()
        vis.run()
        # 明确释放Visualizer资源
        vis.clear_geometries()
        vis.destroy_window()
        # 添加延迟确保资源正确释放
        import time
        time.sleep(0.1)


def create_bbox_lineset(center, size, yaw, color=[1, 0, 0]):
    """
    创建边界框线框，提高可视化稳定性
    Args:
        center: 边界框中心点 [x, y, z]
        size: 边界框尺寸 [width, length, height]
        yaw: 绕Z轴旋转角（弧度）
        color: 线框颜色 [r, g, b]
    Returns:
        o3d.geometry.LineSet 对象
    """
    # 确保所有输入都是浮点数
    center = np.array(center, dtype=np.float64)
    size = np.array(size, dtype=np.float64)
    yaw = float(yaw)
    # 定义旋转矩阵
    R = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ], dtype=np.float64)
    # 定义 bbox corners（局部坐标系下）
    w, l, h = size[0] / 2, size[1] / 2, size[2] / 2
    corners = np.array([
        [-w, -l, -h],
        [w, -l, -h],
        [w, l, -h],
        [-w, l, -h],
        [-w, -l, h],
        [w, -l, h],
        [w, l, h],
        [-w, l, h]
    ], dtype=np.float64)
    # 旋转和平移corners
    corners = (R @ corners.T).T + center
    # 创建线框
    lines = o3d.geometry.LineSet()
    lines.points = o3d.utility.Vector3dVector(corners)
    # 定义线段连接
    lines.lines = o3d.utility.Vector2iVector([
        [0, 1], [1, 2], [2, 3], [3, 0],  # 底部矩形
        [4, 5], [5, 6], [6, 7], [7, 4],  # 顶部矩形
        [0, 4], [1, 5], [2, 6], [3, 7]  # 连接顶部和底部
    ])
    # 设置线框颜色
    lines.colors = o3d.utility.Vector3dVector([color] * len(lines.lines))
    return lines


def parse_bbox(bbox):
    """解析边界框数据，返回中心点、尺寸和偏航角"""
    if isinstance(bbox, dict):
        if 'position' in bbox:
            center = bbox['position']
            size = bbox['dimensions']
            yaw = bbox['rotation']
        else:
            center = [bbox.get('x', 0), bbox.get('y', 0), bbox.get('z', 0)]
            size = [bbox.get('w', 0), bbox.get('l', 0), bbox.get('h', 0)]
            yaw = bbox.get('yaw', 0)
    else:
        center = [bbox[0], bbox[1], bbox[2]]
        size = [bbox[3], bbox[4], bbox[5]]
        yaw = bbox[6]
    # 确保所有值都是浮点数
    center = [float(c) for c in center]
    size = [float(s) for s in size]
    yaw = float(yaw)
    return center, size, yaw


def create_point_cloud_video_h5(hdf5_path, output_video_path, process_frame_func, args, fps=10):
    """
    从HDF5文件创建点云视频函数

    参数:
    - hdf5_path: 输入HDF5文件路径
    - output_video_path: 输出视频文件路径
    - process_frame_func: 处理单帧的函数（返回bboxes和处理后的点云）
    - args: 传递给process_frame_func的参数
    - fps: 视频帧率，默认10
    """
    import h5py

    print("=== Creating Point Cloud Video from HDF5 ===")

    # 确保输出目录存在
    output_dir = os.path.dirname(output_video_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 打开HDF5文件
    with h5py.File(hdf5_path, 'r') as hf:
        # 读取元数据
        columns = list(hf.attrs['columns'])
        args.columns = columns  # 更新args中的columns

        # 读取帧索引
        frame_names = [name.decode('utf-8') if isinstance(name, bytes) else name
                      for name in hf['frame_index/names'][:]]

        print(f"Found {len(frame_names)} frames in HDF5 file")

        # 创建图像对象（只创建一次，重复使用以提高性能）
        fig, ax = plt.subplots(figsize=(10, 10), facecolor='white')
        plt.tight_layout()
        canvas = FigureCanvas(fig)

        # 设置临时帧目录
        temp_frames_dir = os.path.join(output_dir, 'temp_frames')
        if not os.path.exists(temp_frames_dir):
            os.makedirs(temp_frames_dir)

        classes = ['person', 'small vehicle', 'large vehicle', 'other']

        print(f"Processing {len(frame_names)} frames...")
        for i, frame_name in tqdm(enumerate(frame_names)):
            # 从HDF5读取点云数据
            pcd_array = hf[f'frames/{frame_name}'][:]

            # 过滤点云数据
            z_idx = columns.index('z')
            dynamic_idx = columns.index('dynamic_mask')
            filter_mask = (pcd_array[:, z_idx] >= 0.1) & \
                         (pcd_array[:, z_idx] <= 3) & \
                         (pcd_array[:, dynamic_idx] == 1)
            filtered_pcd_array = pcd_array[filter_mask]

            # 处理单帧
            frame_bboxes, processed_pcd_array = process_frame_func(
                i, frame_name, pcd_array, filtered_pcd_array, args
            )

            # 可视化当前帧
            cluster_id = processed_pcd_array[:, -1]  # 假设聚类ID被添加为最后一列
            fig = visualize_frame_for_video(
                processed_pcd_array, cluster_id, frame_bboxes, i, classes, fig, ax
            )

            # 将图转换为图像
            canvas.draw()
            image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

            # 保存帧
            frame_path = os.path.join(temp_frames_dir, f'frame_{i:06d}.jpg')
            cv2.imwrite(frame_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    # 使用OpenCV创建视频
    print("Creating video...")
    frame_sample = cv2.imread(os.path.join(temp_frames_dir, f'frame_{0:06d}.jpg'))
    height, width, _ = frame_sample.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 编码器
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # 将所有帧添加到视频
    for i in tqdm(range(len(frame_names)), desc="Writing video"):
        frame_path = os.path.join(temp_frames_dir, f'frame_{i:06d}.jpg')
        frame = cv2.imread(frame_path)
        video_writer.write(frame)

    # 释放资源
    video_writer.release()
    plt.close(fig)

    # 可选：清理临时帧文件
    import shutil
    shutil.rmtree(temp_frames_dir)

    print(f"Video created successfully: {output_video_path}")
    return output_video_path