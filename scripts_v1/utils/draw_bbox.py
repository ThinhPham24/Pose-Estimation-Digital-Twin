import numpy as np
import os
import cv2
import open3d as o3d
def calculate_2d_projections(coordinates_3d, intrinsics):
    """
    Input: 
        coordinates: [3, N]
        intrinsics: [3, 3]
    Return 
        projected_coordinates: [N, 2]
    """
    projected_coordinates = intrinsics @ coordinates_3d
    projected_coordinates = projected_coordinates[:2, :] / projected_coordinates[2, :]
    projected_coordinates = projected_coordinates.transpose()
    
    projected_coordinates = np.array(projected_coordinates, dtype=np.int32)
    return projected_coordinates
def draw_axis(img, R, t, K, axis_length=0.05):
    """
    Draws a coordinate frame on the image.

    Parameters:
        img: The image to draw on.
        R: (3,3) Rotation matrix from object to camera.
        t: (3,) Translation vector.
        K: (3,3) Camera intrinsics.
        axis_length: Length of the axis in meters.
    """
    # Define axis in object coordinate frame
    axes = np.array([
        [0,0,0],              # origin
        [axis_length, 0, 0],  # x
        [0, axis_length, 0],  # y
        [0, 0, axis_length]   # z
    ]).transpose()  # shape: (3, 4)

    # Transform to camera frame: X_cam = R @ X_obj + t
    axes_cam = R @ axes + t[:, np.newaxis]

    # Project to 2D
    proj = K @ axes_cam
    proj = proj[:2] / proj[2]
    proj = proj.transpose()
    proj = np.array(proj, dtype=np.int32)

    o, x, y, z = proj
    img = cv2.arrowedLine(img, tuple(o), tuple(x), (0, 0, 255), 2)   # X - blue
    img = cv2.arrowedLine(img, tuple(o), tuple(y), (0, 255, 0), 2)   # Y - green
    img = cv2.arrowedLine(img, tuple(o), tuple(z), (255, 0, 0), 2)   # Z - Red

    return img
def get_3d_bbox(scale, shift = 0):
    """
    Input: 
        scale: [3] or scalar
        shift: [3] or scalar
    Return 
        bbox_3d: [3, N]

    """
    if hasattr(scale, "__iter__"):
        bbox_3d = np.array([[scale[0] / 2, +scale[1] / 2, scale[2] / 2],
                  [scale[0] / 2, +scale[1] / 2, -scale[2] / 2],
                  [-scale[0] / 2, +scale[1] / 2, scale[2] / 2],
                  [-scale[0] / 2, +scale[1] / 2, -scale[2] / 2],
                  [+scale[0] / 2, -scale[1] / 2, scale[2] / 2],
                  [+scale[0] / 2, -scale[1] / 2, -scale[2] / 2],
                  [-scale[0] / 2, -scale[1] / 2, scale[2] / 2],
                  [-scale[0] / 2, -scale[1] / 2, -scale[2] / 2]]) + shift
    else:
        bbox_3d = np.array([[scale / 2, +scale / 2, scale / 2],
                  [scale / 2, +scale / 2, -scale / 2],
                  [-scale / 2, +scale / 2, scale / 2],
                  [-scale / 2, +scale / 2, -scale / 2],
                  [+scale / 2, -scale / 2, scale / 2],
                  [+scale / 2, -scale / 2, -scale / 2],
                  [-scale / 2, -scale / 2, scale / 2],
                  [-scale / 2, -scale / 2, -scale / 2]]) + shift
    bbox_3d = bbox_3d.transpose()
    return bbox_3d

def draw_3d_bbox(img, imgpts, color, size=3):
    imgpts = np.int32(imgpts).reshape(-1, 2)

    # draw ground layer in darker color
    color_ground = (int(color[0] * 0.3), int(color[1] * 0.3), int(color[2] * 0.3))
    for i, j in zip([4, 5, 6, 7],[5, 7, 4, 6]):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), color_ground, size)

    # draw pillars in blue color
    color_pillar = (int(color[0]*0.6), int(color[1]*0.6), int(color[2]*0.6))
    for i, j in zip(range(4),range(4,8)):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), color_pillar, size)

    # finally, draw top layer in color
    for i, j in zip([0, 1, 2, 3],[1, 3, 0, 2]):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), color, size)
    return img

def draw_3d_pts(img, imgpts, color, size=1):
    imgpts = np.int32(imgpts).reshape(-1, 2)
    for point in imgpts:
        img = cv2.circle(img, (point[0], point[1]), size, color, -1)
    return img

def draw_detections(image,pred_rots, pred_trans, model_points, intrinsics, color=(255, 0, 0)):
    """
    Draws 3D bounding boxes and projected point clouds onto an image.

    Args:
        image (np.ndarray): The 2D image on which to draw.
        pred_rots (list of np.ndarray or np.ndarray): List of (3,3) rotation matrices.
        pred_trans (list of np.ndarray or np.ndarray): List of (3,) translation vectors.
        model_points (np.ndarray): Nx3 array of 3D points representing the object.
        intrinsics (np.ndarray): (3,3) camera intrinsic matrix.
        color (tuple): RGB color for drawing.
    
    Returns:
        np.ndarray: Image with drawn 3D bounding boxes.
    """

    num_pred_instances = len(pred_rots)
    draw_image_bbox = image.copy()

    # Compute scale and shift for the bounding box
    scale = (np.max(model_points, axis=0) - np.min(model_points, axis=0))
    shift = np.mean(model_points, axis=0)
    bbox_3d = get_3d_bbox(scale, shift)

    # Sample 3D points from the model
    choose = np.random.choice(np.arange(len(model_points)), 512, replace=False)
    pts_3d = model_points[choose].T  # Shape (3, 512)

    for ind in range(num_pred_instances):
        rot_matrix = pred_rots[ind]  # (3,3)
        trans_vector = pred_trans[ind]  # (3,)
        # Transform 3D bounding box
        transformed_bbox_3d = rot_matrix @ bbox_3d + trans_vector[:, np.newaxis]
        projected_bbox = calculate_2d_projections(transformed_bbox_3d, intrinsics)
        draw_image_bbox = draw_3d_bbox(draw_image_bbox, projected_bbox, color)

        # Transform point cloud
        transformed_pts_3d = rot_matrix @ pts_3d + trans_vector[:, np.newaxis]
        projected_pts = calculate_2d_projections(transformed_pts_3d, intrinsics)
        draw_image_bbox = draw_3d_pts(draw_image_bbox, projected_pts, color)
        # ✅ Draw 3D coordinate axes
        draw_image_bbox = draw_axis(draw_image_bbox, rot_matrix, trans_vector, intrinsics)
    return draw_image_bbox
def get_3d_bbox_from_point_cloud(point_cloud):
    """
    Computes the 3D bounding box of a given point cloud.

    Args:
        point_cloud (np.ndarray): Nx3 array of 3D points.

    Returns:
        bbox_corners (np.ndarray): 8x3 array of bounding box corner points.
    """
    min_corner = np.min(point_cloud, axis=0)
    max_corner = np.max(point_cloud, axis=0)

    bbox_corners = np.array([
        [min_corner[0], min_corner[1], min_corner[2]],
        [min_corner[0], min_corner[1], max_corner[2]],
        [min_corner[0], max_corner[1], min_corner[2]],
        [min_corner[0], max_corner[1], max_corner[2]],
        [max_corner[0], min_corner[1], min_corner[2]],
        [max_corner[0], min_corner[1], max_corner[2]],
        [max_corner[0], max_corner[1], min_corner[2]],
        [max_corner[0], max_corner[1], max_corner[2]],
    ])
    return bbox_corners

def transform_bbox(bbox_corners, transformation_matrix):
    """
    Applies a transformation matrix to the bounding box.

    Args:
        bbox_corners (np.ndarray): 8x3 array of bounding box corner points.
        transformation_matrix (np.ndarray): 4x4 transformation matrix.

    Returns:
        transformed_bbox (np.ndarray): 8x3 transformed bounding box.
    """
    bbox_homogeneous = np.hstack((bbox_corners, np.ones((bbox_corners.shape[0], 1))))  # Convert to homogeneous coordinates
    transformed_bbox = (transformation_matrix @ bbox_homogeneous.T).T[:, :3]  # Apply transformation and convert back
    return transformed_bbox

def draw_3d_bbox_on_point_cloud(point_cloud, scens, transformation_matrix=None):
    """
    Visualizes the point cloud with a 3D bounding box.

    Args:
        point_cloud (np.ndarray): Nx3 array of 3D points.
        transformation_matrix (np.ndarray, optional): 4x4 transformation matrix.

    Returns:
        None
    """
    # Convert point cloud to Open3D format
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)

    # Compute 3D bounding box
    bbox_corners = get_3d_bbox_from_point_cloud(point_cloud)
    print("transformation_matrix", transformation_matrix)
    inverse = np.linalg.inv(transformation_matrix)
    print("Inverse", inverse)
    # Apply transformation if provided
    if transformation_matrix is not None:
        bbox_corners = transform_bbox(bbox_corners, inverse)

    # Create lines connecting the bounding box corners
    lines = [
        [0, 1], [0, 2], [0, 4], [1, 3], [1, 5], [2, 3],
        [2, 6], [3, 7], [4, 5], [4, 6], [5, 7], [6, 7]
    ]

    # Create bounding box edges
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(bbox_corners)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector([[1, 0, 0] for _ in lines])  # Red color for the bounding box
    source_temp = pcd.transform(inverse)
    # Visualize the point cloud and bounding box
    o3d.visualization.draw_geometries([source_temp, scens, line_set])