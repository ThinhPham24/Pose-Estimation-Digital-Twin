# import cv2
# import numpy as np
# import sys
# import open3d as o3d
# from cv2 import ppf_match_3d as ppf
# def scale_point_cloud(point_cloud, scale_factor=0.001):
#     """
#     Scales the given Open3D point cloud by the specified factor.
#     Default: Converts mm to meters by dividing by 1000.
#     """
#     point_cloud= np.asarray(point_cloud[:3]) / scale_factor
#     return point_cloud
# def preprocess_point_cloud(pcd, voxel_size):
#     """ Downsample and estimate normals for feature extraction. """
#     pcd_down = pcd.voxel_down_sample(voxel_size)
#     pcd_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=100))
#     pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
#         pcd_down, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=50)
#     )
#     return pcd_down, pcd_fpfh
# model_file = R"/home/airlab/Desktop/DigitalTwin_PoseEstimation/data/ply_models/obj1.ply"
# # scene_file = R"/home/airlab/Desktop/DigitalTwin_PoseEstimation/data/source/d435_on_ur10e_arm/source_ob1_3.ply"
# source = o3d.io.read_point_cloud(R"/home/airlab/Desktop/DigitalTwin_PoseEstimation/data/source/d435_on_ur10e_arm/source_ob1_0.ply")
# target = o3d.io.read_point_cloud(R"/home/airlab/Desktop/DigitalTwin_PoseEstimation/data/ply_models/obj1.ply")
# # Load the model and scene point clouds
# print("Loading 3D model and scene...")
# pc_model = ppf.loadPLYSimple(model_file, 1)  # Load model (1 = with normals)
# print("pc_model", pc_model)
# # pc_scene = ppf.loadPLYSimple(scene_file, 0)  # Load scene (1 = with normals)
# # target = scale_point_cloud(target)
# # pc_model, _ = preprocess_point_cloud(target, 0.01)
# # pc_model = np.asarray(np.stack((pc_model.points,pc_model.normals),axis =-1)).astype(np.float32)
# # print("pc_model", pc_model)
# pc_model = scale_point_cloud(pc_model)
# print("scale pc_model", pc_model)
# # pc_scene, _ = preprocess_point_cloud(pc_scene, 0.1)
# # pc_scene = np.asarray(pc_scene.points).astype(np.float32)
# # print("pc_scene", pc_scene)
# # Train the model
# detector = ppf.PPF3DDetector(0.02, 0.05)
# print("Training model...")
# t_start = cv2.getTickCount()
# detector.trainModel(pc_model)
# t_end = cv2.getTickCount()
# print(f"Training completed in {(t_end - t_start) / cv2.getTickFrequency():.2f} seconds")

# # Perform matching
# print("Matching model to scene...")
# t_start = cv2.getTickCount()
# results = detector.match(pc_scene, relativeSceneSampleStep=1/5, relativeDistanceThreshold=0.01)
# t_end = cv2.getTickCount()
# print(f"Matching completed in {(t_end - t_start) / cv2.getTickFrequency():.2f} seconds")

# # Take top N results
# N = min(2, len(results))
# top_results = results[:N]

# # Perform ICP to refine the pose
# icp = ppf.ICP(iterations=100, tolerances=0.005, rejectionScale=2.5, numLevels=8)
# print(f"Performing ICP on {N} poses...")
# t_start = cv2.getTickCount()
# icp.registerModelToScene(pc_model, pc_scene, top_results)
# t_end = cv2.getTickCount()
# print(f"ICP completed in {(t_end - t_start) / cv2.getTickFrequency():.2f} seconds")

# # Print the refined poses
# print("Final refined poses:")
# for i, pose in enumerate(top_results):
#     print(f"Pose {i+1}:")
#     pose.printPose()
#     # Save the transformed model for visualization
#     if i == 0:
#         transformed_pc = ppf.transformPCPose(pc_model, pose.pose)
#         ppf.writePLY(transformed_pc, "transformed_model.ply")
#         print("Transformed model saved as transformed_model.ply")

# import cv2 
# import open3d as o3d
# import numpy as np
# def scale_point_cloud(point_cloud, scale_factor=1000.0):
#     """
#     Scales the given Open3D point cloud by the specified factor.
#     Default: Converts mm to meters by dividing by 1000.
#     """
#     point_cloud.points = o3d.utility.Vector3dVector(np.asarray(point_cloud.points) / scale_factor)
#     return point_cloud
# def paint_point_cloud(pcd , color):
#     """Paints a point cloud with a specified color using Open3D."""
#     # Convert color to RGB (0-1 range)
#     color_rgb = np.array(color) / 255.0

#     # Create an array of colors for each point
#     colors = np.tile(color_rgb, (len(pcd.points), 1))

#     # Assign colors to the point cloud
#     pcd.colors = o3d.utility.Vector3dVector(colors)
#     return pcd
# N = 2
# modelname = "/home/airlab/Desktop/DigitalTwin_PoseEstimation/data/ply_models/obj1.ply"
# # scenename = "/home/airlab/Desktop/DigitalTwin_PoseEstimation/data/ply_models/obj1.ply"
# # modelname = "/home/airlab/Desktop/DigitalTwin_PoseEstimation/data/source/d435_on_ur10e_arm/source_ob1_0.ply"
# scenename = "/home/airlab/Desktop/DigitalTwin_PoseEstimation/data/source/d435_on_ur10e_arm/source_ob1_0.ply"

# detector = cv2.ppf_match_3d_PPF3DDetector(0.01, 0.001)

# print('Loading model...')
# pc = cv2.ppf_match_3d.loadPLYSimple(modelname, 1)
# print("pOINT LCOUD", pc)

# print('Training...')
# detector.trainModel(pc)

# print('Loading scene...')
# pcTest = cv2.ppf_match_3d.loadPLYSimple(scenename, 1)
# # pcTest[:, :3] /= 1000 
# print('Matching...')
# results = detector.match(pcTest,0.001, 1)

# print('Performing ICP...')
# icp = cv2.ppf_match_3d_ICP(100)
# _, results = icp.registerModelToScene(pc, pcTest, results[:N])

# print("Poses: ")
# for i, result in enumerate(results):
#     #result.printPose()
#     print("\n-- Pose to Model Index %d: NumVotes = %d, Residual = %f\n%s\n" % (result.modelIndex, result.numVotes, result.residual, result.pose))
#     if i == 0:
#         pct = cv2.ppf_match_3d.transformPCPose(pc, result.pose)
#         modelname = o3d.io.read_point_cloud(modelname)
#         scenename = o3d.io.read_point_cloud(scenename)
#         color = [255, 0, 0]  # Red color (RGB)
#         modelname = paint_point_cloud(modelname,color)
#         # scale = scale_point_cloud(scenename)
#         o3d.visualization.draw_geometries([modelname.transform(result.pose), scenename])
#         cv2.ppf_match_3d.writePLY(pct, "PCTrans1.ply")
#         print("THINH")
#         # for i, pose in enumerate(top_results):
# #     print(f"Pose {i+1}:")
# #     pose.printPose()
# #     # Save the transformed model for visualization
# #     if i == 0:
# #         transformed_pc = ppf.transformPCPose(pc_model, pose.pose)
# #         ppf.writePLY(transformed_pc, "transformed_model.ply")
# #         print("Transformed model saved as transformed_model.ply")
import numpy as np
import open3d as o3d
from sklearn.neighbors import KDTree

def compute_pair_point_feature(point_cloud, radius):
    """
    Computes Pair Point Features (PPF) for a point cloud.

    Args:
        point_cloud (numpy.ndarray): Nx3 array representing the point cloud.
        radius (float): The radius to search for neighboring points.

    Returns:
        numpy.ndarray: Nx4 array of PPF features.
    """
    num_points = point_cloud.shape[0]
    ppf_features = np.zeros((num_points, 4))
    kdtree = KDTree(point_cloud)

    for i in range(num_points):
        p1 = point_cloud[i]
        neighbors_indices = kdtree.query_radius([p1], r=radius)[0]
        neighbors_indices = neighbors_indices[neighbors_indices != i] #Exclude the point itself.

        if len(neighbors_indices) == 0:
            ppf_features[i] = [0,0,0,0] # Handle isolated points.
            continue

        p2 = point_cloud[neighbors_indices[0]] # Get the first neighbor.

        d = np.linalg.norm(p2 - p1)
        n1 = compute_normal(point_cloud, kdtree, i, radius)
        n2 = compute_normal(point_cloud, kdtree, neighbors_indices[0], radius)

        if np.linalg.norm(n1) < 1e-8 or np.linalg.norm(n2) < 1e-8: #Avoid division by zero
          ppf_features[i] = [0,0,0,0]
          continue

        alpha = np.arccos(np.dot(n1, (p2 - p1) / d))
        phi = np.arccos(np.dot(n2, (p2 - p1) / d))
        delta = np.arccos(np.dot(n1, n2))

        ppf_features[i] = [d, alpha, phi, delta]

    return ppf_features

def compute_normal(point_cloud, kdtree, index, radius):
    """
    Computes the normal vector for a point.

    Args:
        point_cloud (numpy.ndarray): Nx3 array representing the point cloud.
        kdtree (KDTree): KDTree for efficient neighbor search.
        index (int): Index of the point to compute the normal for.
        radius (float): Radius to search for neighbors.

    Returns:
        numpy.ndarray: 3D normal vector.
    """
    neighbors_indices = kdtree.query_radius([point_cloud[index]], r=radius)[0]
    neighbors = point_cloud[neighbors_indices]

    if neighbors.shape[0] < 3:
        return np.array([0, 0, 0])  # Handle cases with insufficient neighbors.

    centroid = np.mean(neighbors, axis=0)
    centered_neighbors = neighbors - centroid
    covariance_matrix = np.cov(centered_neighbors.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    normal = eigenvectors[:, np.argmin(eigenvalues)]
    return normal

def match_point_clouds_ppf(source_cloud, target_cloud, radius, threshold):
    """
    Matches two point clouds using Pair Point Features.

    Args:
        source_cloud (numpy.ndarray): Nx3 array representing the source point cloud.
        target_cloud (numpy.ndarray): Mx3 array representing the target point cloud.
        radius (float): Radius for PPF computation.
        threshold (float): Threshold for feature matching.

    Returns:
        list: List of matched point indices (source_indices, target_indices).
    """

    source_features = compute_pair_point_feature(source_cloud, radius)
    target_features = compute_pair_point_feature(target_cloud, radius)

    source_kdtree = KDTree(source_features)
    distances, target_indices = source_kdtree.query(target_features)

    matched_indices = []
    for i, dist in enumerate(distances):
        if dist[0] < threshold:
            matched_indices.append((target_indices[i][0], i))

    source_indices = [x[0] for x in matched_indices]
    target_indices = [x[1] for x in matched_indices]

    return source_indices, target_indices

def load_ply(filepath):
    """Loads a PLY file and returns a numpy array of points."""
    pcd = o3d.io.read_point_cloud(filepath)
    return np.asarray(pcd.points)

def refine_with_icp(source_pcd, target_pcd, initial_transform=None):
    """Refines the matching using ICP."""
    if initial_transform is None:
        result_icp = o3d.pipelines.registration.registration_icp(
            source_pcd, target_pcd, 0.02, np.identity(4),
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000))
    else:
        result_icp = o3d.pipelines.registration.registration_icp(
            source_pcd, target_pcd, 0.02, initial_transform,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000))

    return result_icp.transformation

def visualize_registration(source_pcd, target_pcd, transformation):
    """Visualizes the registration result."""
    source_temp = copy.deepcopy(source_pcd)
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_pcd])

import copy

# Example Usage with PLY files and ICP refinement:
# modelname = "/home/airlab/Desktop/DigitalTwin_PoseEstimation/data/ply_models/obj1.ply"
# # scenename = "/home/airlab/Desktop/DigitalTwin_PoseEstimation/data/ply_models/obj1.ply"
# # modelname = "/home/airlab/Desktop/DigitalTwin_PoseEstimation/data/source/d435_on_ur10e_arm/source_ob1_0.ply"
# scenename = "/home/airlab/Desktop/DigitalTwin_PoseEstimation/data/source/d435_on_ur10e_arm/source_ob1_0.ply"
source_file = "/home/airlab/Desktop/DigitalTwin_PoseEstimation/data/source/d435_on_ur10e_arm/source_ob1_0.ply"
target_file = "/home/airlab/Desktop/DigitalTwin_PoseEstimation/scripts/partial_point_clouds/view4.ply"

try:
    source_points = load_ply(source_file)
    target_points = load_ply(target_file)

    radius = 0.1
    threshold = 0.15

    source_indices, target_indices = match_point_clouds_ppf(source_points, target_points, radius, threshold)

    source_pcd = o3d.geometry.PointCloud()
    source_pcd.points = o3d.utility.Vector3dVector(source_points)
    target_pcd = o3d.geometry.PointCloud()
    target_pcd.points = o3d.utility.Vector3dVector(target_points)

    # Estimate an initial transformation based on matched points.
    if len(source_indices) > 3: #Need at least 3 points to create a good transform
        source_matched_points = source_points[source_indices]
        target_matched_points = target_points[target_indices]

        # Calculate the transformation using the matched points directly.
        source_centroid = np.mean(source_matched_points, axis=0)
        target_centroid = np.mean(target_matched_points, axis=0)
        H = np.dot((source_matched_points - source_centroid).T, (target_matched_points - target_centroid))
        U, S, Vt = np.linalg.svd(H)
        R = np.dot(Vt.T, U.T)
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = np.dot(Vt.T, U.T)
        t = target_centroid - np.dot(R, source_centroid)
        initial_transform = np.eye(4)
        initial_transform[:3, :3] = R
        initial_transform[:3, 3] = t

        # Refine with ICP.
        final_transform = refine_with_icp(source_pcd, target_pcd, initial_transform)

        # Visualize the result.
        visualize_registration(source_pcd, target_pcd, final_transform)

    else:
        print("Insufficient matches for ICP refinement.")

except FileNotFoundError:
    print("Error: One or both PLY files not found.")
except Exception as e:
    print(f"An error occurred: {e}")