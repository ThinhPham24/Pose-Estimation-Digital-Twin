# import os, sys, time,torch,pickle,trimesh,itertools,pdb,zipfile,datetime,imageio,gzip,logging,joblib,importlib,uuid,signal,multiprocessing,psutil,subprocess,tarfile,scipy,argparse
# from pytorch3d.transforms import so3_log_map,so3_exp_map,se3_exp_map,se3_log_map,matrix_to_axis_angle,matrix_to_euler_angles,euler_angles_to_matrix, rotation_6d_to_matrix
# from pytorch3d.renderer import FoVPerspectiveCameras, PerspectiveCameras, look_at_view_transform, look_at_rotation, RasterizationSettings, MeshRenderer, MeshRasterizer, BlendParams, SoftSilhouetteShader, HardPhongShader, PointLights, TexturesVertex
# from pytorch3d.renderer.mesh.rasterize_meshes import barycentric_coordinates
# from pytorch3d.renderer.mesh.shader import SoftDepthShader, HardFlatShader
# from pytorch3d.renderer.mesh.textures import Textures
# from pytorch3d.structures import Meshes
# from scipy.interpolate import griddata
# import nvdiffrast.torch as dr
# import torch.nn.functional as F
# import torchvision
# import torch.nn as nn
# from functools import partial
# import pandas as pd
# import open3d as o3d
# from uuid import uuid4
# import cv2
# from PIL import Image
import numpy as np
# from collections import defaultdict
# import multiprocessing as mp
# import matplotlib.pyplot as plt
# import math,glob,re,copy
# from transformations import *
# from scipy.spatial import cKDTree
# from collections import OrderedDict
# def compute_mesh_diameter(model_pts=None, mesh=None, n_sample=1000):
#   from sklearn.decomposition import TruncatedSVD
#   if mesh is not None:
#     u, s, vh = scipy.linalg.svd(mesh.vertices, full_matrices=False)
#     pts = u@s
#     diameter = np.linalg.norm(pts.max(axis=0)-pts.min(axis=0))
#     return float(diameter)

#   if n_sample is None:
#     pts = model_pts
#   else:
#     ids = np.random.choice(len(model_pts), size=min(n_sample, len(model_pts)), replace=False)
#     pts = model_pts[ids]
#   dists = np.linalg.norm(pts[None]-pts[:,None], axis=-1)
#   diameter = dists.max()
#   return diameter
def project_points_3d_to_2d(points_3d, colors, K):
    """
    Projects 3D points to 2D and assigns colors based on the 3D points.

    Args:
    - points_3d (np.ndarray): Nx3 array of 3D points [X, Y, Z]
    - colors (np.ndarray): Nx3 array of RGB color values corresponding to each 3D point
    - K (np.ndarray): 3x3 camera intrinsic matrix

    Returns:
    - np.ndarray: 2D image with colored points projected from 3D space
    """
    # Ensure points are Nx3 (non-homogeneous)
    if points_3d.shape[1] != 3:
        raise ValueError(f"Expected points_3d with shape (N, 3), got {points_3d.shape}")
    # Add a row of ones to make points homogeneous
    points_3d_hom = np.hstack((points_3d, np.ones((points_3d.shape[0], 1))))

    # Project points using the camera intrinsic matrix
    points_2d_hom = K @ points_3d_hom[:, :3].T  # Use only the first 3 columns (Nx3)
    
    # Normalize the homogeneous coordinates
    points_2d = points_2d_hom[:2, :] / points_2d_hom[2, :]  # Divide by z to normalize
    
    # Convert to integer pixel coordinates
    points_2d = np.round(points_2d.T).astype(np.int32)

    # Initialize a blank image to draw the points on (black background)
    image = np.zeros((480, 640, 3), dtype=np.uint8)

    # Loop over each point and its color
    for i, point in enumerate(points_2d):
        u, v = point
        if 0 <= u < image.shape[1] and 0 <= v < image.shape[0]:  # Make sure it's within image bounds
            image[v, u] = colors[i]  # Set the color at the projected point

    return image