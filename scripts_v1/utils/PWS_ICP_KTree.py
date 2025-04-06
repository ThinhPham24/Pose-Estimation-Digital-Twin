import numpy as np
import open3d as o3d
from scipy.spatial import ConvexHull

# ----------------------- Utility Functions -----------------------
def scale_point_cloud(point_cloud, scale_factor= 1000):
    """
    Scales the given Open3D point cloud by the specified factor.
    Default: Converts mm to meters by dividing by 1000.
    """
    point_cloud.points = o3d.utility.Vector3dVector(np.asarray(point_cloud.points) / scale_factor)
    return point_cloud

def preprocess_point_cloud(pcd, voxel_size):
    """ Downsample and estimate normals for feature extraction. """
    pcd_down = pcd.voxel_down_sample(voxel_size)
    pcd_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=10))
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down, o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=50)
    )
    return pcd_down, pcd_fpfh
def pairwise_feature_matching(source_fpfh, target_fpfh):
    """Find nearest neighbor matches between two feature sets."""
    source_features = np.asarray(source_fpfh.data).T
    target_features = np.asarray(target_fpfh.data).T

    # Compute nearest neighbors
    from scipy.spatial import cKDTree
    target_tree = cKDTree(target_features)
    distances, indices = target_tree.query(source_features, k=1)  # Nearest neighbor

    # Convert to Open3D CorrespondenceSet
    correspondences = np.vstack((np.arange(len(indices)), indices.flatten())).T
    return correspondences
def execute_ransac(source_down, target_down, correspondences, voxel_size):
    """Perform transformation estimation using RANSAC on pairwise correspondences."""
    distance_threshold = voxel_size * 1.5  # RANSAC threshold

    result = o3d.pipelines.registration.registration_ransac_based_on_correspondence(
        source_down, target_down, o3d.utility.Vector2iVector(correspondences),
        max_correspondence_distance=distance_threshold,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        ransac_n= 5,  # Number of points per RANSAC iteration
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999)
    )
    return result
def refine_with_icp(source, target, initial_transform, voxel_size):
    """Refine alignment using ICP."""
    max_correspondence_distance = voxel_size * 0.4  # ICP threshold
    result_icp = o3d.pipelines.registration.registration_icp(
        source, target, max_correspondence_distance, initial_transform,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=1000)
    )
    return result_icp
def execute_global_ransac_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    print(":: RANSAC registration on downsampled point clouds.")
    print("   Since the downsampling voxel size is %.3f," % voxel_size)
    print("   we use a liberal distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    return result

def draw_registration_result(source, target, transformation):
    """ Draw source and target point clouds with transformation. """
    source_temp = source.transform(transformation)
    source_temp.paint_uniform_color([1, 0, 0])  # Red
    target.paint_uniform_color([0, 1, 0])  # Green
    o3d.visualization.draw_geometries([source_temp, target])

def seg_plane(pcd):
    """ Segment the dominant plane using RANSAC. """
    plane_model, inliers = pcd.segment_plane(distance_threshold=0.002, ransac_n=3, num_iterations=100)
    inlier_cloud = pcd.select_by_index(inliers)  # Plane points
    outlier_cloud = pcd.select_by_index(inliers, invert=True)  # Objects
    return inlier_cloud, outlier_cloud

def remove_outliers(pcd, nb_neighbors=10, std_ratio=2.0, nb_points=5, radius=0.02):
    """ Remove statistical and radius outliers. """
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    inlier_cloud = pcd.select_by_index(ind)
    cl_2, ind_2 = inlier_cloud.remove_radius_outlier(nb_points=nb_points, radius=radius)
    return cl_2

# ----------------------- Main Pipeline -----------------------
if __name__ == "__main__":
    try:
        # Load Point Clouds
        source = o3d.io.read_point_cloud("/home/airlab/Desktop/DigitalTwin_PoseEstimation/data/source/d435_on_ur10e_arm/source_ob1_0.ply")
        target = o3d.io.read_point_cloud('/home/airlab/Desktop/DigitalTwin_PoseEstimation/scripts/partial_point_clouds/view2.ply')
        init_target = o3d.io.read_point_cloud('/home/airlab/Desktop/DigitalTwin_PoseEstimation/data/ply_models/obj1.ply')
        # Scale to meters
        outlier_source = source
        # Remove noise and outliers
        # outlier_source = remove_outliers(outlier_source)
        # outlier_target = remove_outliers(target)
        outlier_target = target
        o3d.visualization.draw_geometries([outlier_source, outlier_target])

        # # Downsample for efficiency
        voxel_size_source = 0.005
        voxel_size_target = 0.005
        source_down, source_fpfh = preprocess_point_cloud(outlier_source, voxel_size_source)
        target_down, target_fpfh = preprocess_point_cloud(outlier_target, voxel_size_target)
        o3d.visualization.draw_geometries([source_down, target_down])
        # Pairwise feature matching
        correspondences = pairwise_feature_matching(source_fpfh, target_fpfh)
        voxel_size = 0.0025
        # Estimate transformation using RANSAC
        ransac_result = execute_ransac(source_down, target_down, correspondences, voxel_size)

        # Refine with ICP
        icp_result = refine_with_icp(outlier_source, outlier_target, ransac_result.transformation, voxel_size)

        # Visualize final alignment
        o3d.visualization.draw_geometries([outlier_source.transform(icp_result.transformation), outlier_target])
        o3d.visualization.draw_geometries([outlier_source, init_target])
        print("Transformation matrix:", icp_result.transformation)
        

    except Exception as e:
        print("Error:", e)