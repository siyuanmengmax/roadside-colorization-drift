# utils.py

import open3d as o3d
import pickle
import os
import numpy as np

def ensure_directory(directory):
    """
    Ensure that a directory exists.
    :param directory:
    :return:
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f'Created directory: {directory}')

def create_pcd(pcd_array, downsample=False, voxel_size=1.0, estimate_normals=False, fpfh=False, colored=False,
               columns=['x', 'y', 'z', 'range', 'reflectivity', 'near_ir', 'intensity','flags']):
    """
    Create an Open3D point cloud from a numpy array.
    :param pcd_array: numpy array of shape (N, M) where N is the number of points and M is the number of features
    :param downsample: whether to downsample the point cloud
    :param voxel_size: voxel size for downsampling
    :param estimate_normals: whether to estimate normals
    :param fpfh: whether to compute FPFH features
    :return: Open3D point cloud
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcd_array[:, :3])
    if colored:
        colors = np.zeros((pcd_array.shape[0], 3))
        reflectivity = pcd_array[:, columns.index('reflectivity')]
        colors[:, 0] = reflectivity / 255.0
        near_ir = pcd_array[:, columns.index('near_ir')]
        colors[:, 2] = np.clip(near_ir / 592, 0, 1)
        colors[:, 1] = 0
        pcd.colors = o3d.utility.Vector3dVector(colors)
    if downsample:
        pcd = pcd.voxel_down_sample(voxel_size)
        if estimate_normals:
            radius_normal = voxel_size * 2
            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
            if fpfh:
                radius_feature = voxel_size * 5
                pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
                    pcd, search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
                return pcd, pcd_fpfh
    return pcd

def read_pkl(file_path):
    """
    Read a pickle file.
    :param file_path:
    :return:
    """
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data

def write_pkl(file_path, data):
    """
    Write a pickle file.
    :param file_path:
    :param data:
    :return:
    """
    with open(file_path, 'wb') as file:
        pickle.dump(data, file)
    print(f'Saved data to {file_path}')

def write_bin(points_array, output_path):
    """
    将点云数组保存为二进制文件(.bin)

    参数:
    points_array: numpy array, 点云数据
    output_path: str, 输出文件路径
    """
    # 确保输出目录存在
    # os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # 将数组转换为float32类型(如果还不是)并保存为二进制文件
    points_array = points_array.astype(np.float32)
    points_array.tofile(output_path)

def read_bin(bin_path, num_features):
    """
    读取二进制格式的点云文件(.bin)

    参数:
    bin_path: str, bin文件路径
    num_features: int, 可选，每个点的特征维度。如果为None，将自动推断

    返回:
    points_array: numpy array, 形状为(N, num_features)的点云数组
    """
    # 读取原始二进制数据
    points_array = np.fromfile(bin_path, dtype=np.float32)
    # 重塑数组
    points = points_array.reshape(-1, num_features) # -1表示自动计算行数，指定列数为num_features
    return points

def transform_pcd(points, transformation_matrix):
    """Transform point cloud using given transformation matrix"""
    # Add homogeneous coordinate
    homogeneous = np.column_stack([points[:, :3], np.ones((len(points),1))])
    transformed = (transformation_matrix @ homogeneous.T).T
    transformed = np.column_stack([transformed[:,:3], points[:, 3:]])
    return transformed
