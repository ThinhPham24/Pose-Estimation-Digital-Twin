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

import cv2 
import open3d as o3d
import numpy as np
def scale_point_cloud(point_cloud, scale_factor=1000.0):
    """
    Scales the given Open3D point cloud by the specified factor.
    Default: Converts mm to meters by dividing by 1000.
    """
    point_cloud.points = o3d.utility.Vector3dVector(np.asarray(point_cloud.points) / scale_factor)
    return point_cloud
N = 5
# modelname = "/home/airlab/Desktop/DigitalTwin_PoseEstimation/data/ply_models/obj1.ply"
modelname = "/home/airlab/Desktop/DigitalTwin_PoseEstimation/data/source/d435_on_ur10e_arm/source_ob1_0.ply"
scenename = "/home/airlab/Desktop/DigitalTwin_PoseEstimation/data/source/d435_on_ur10e_arm/source_ob1_0.ply"

detector = cv2.ppf_match_3d_PPF3DDetector(0.001, 0.0001)

print('Loading model...')
pc = cv2.ppf_match_3d.loadPLYSimple(modelname, 1)
print("pOINT LCOUD", pc)

print('Training...')
detector.trainModel(pc)

print('Loading scene...')
pcTest = cv2.ppf_match_3d.loadPLYSimple(scenename, 1)
# pcTest[:, :3] /= 1000 
print('Matching...')
results = detector.match(pcTest, 1, 0.001)

print('Performing ICP...')
icp = cv2.ppf_match_3d_ICP(100)
_, results = icp.registerModelToScene(pc, pcTest, results[:N])

print("Poses: ")
for i, result in enumerate(results):
    #result.printPose()
    print("\n-- Pose to Model Index %d: NumVotes = %d, Residual = %f\n%s\n" % (result.modelIndex, result.numVotes, result.residual, result.pose))
    if i == 0:
        pct = cv2.ppf_match_3d.transformPCPose(pc, result.pose)
        modelname = o3d.io.read_point_cloud(modelname)
        scenename = o3d.io.read_point_cloud(scenename)
        # scale = scale_point_cloud(scenename)
        o3d.visualization.draw_geometries([modelname.transform(result.pose), scenename])
        cv2.ppf_match_3d.writePLY(pct, "PCTrans1.ply")
        print("THINH")
        # for i, pose in enumerate(top_results):
#     print(f"Pose {i+1}:")
#     pose.printPose()
#     # Save the transformed model for visualization
#     if i == 0:
#         transformed_pc = ppf.transformPCPose(pc_model, pose.pose)
#         ppf.writePLY(transformed_pc, "transformed_model.ply")
#         print("Transformed model saved as transformed_model.ply")