import numpy as np
import open3d as o3d
from scipy.spatial import ConvexHull

class PointCloudRegistrator:
    def __init__(self, source_path, target_path, init_target_path=None):
        self.source_path = source_path
        self.target_path = target_path
        self.init_target_path = init_target_path
        self.source = None
        self.target = None
        self.init_target = None
        self.voxel_size_source = 0.005
        self.voxel_size_target = 0.005
        self.voxel_size = 0.0025

    def load_point_clouds(self):
        try:
            # self.source = o3d.io.read_point_cloud(self.source_path)
            self.source = self.source_path
            self.target = o3d.io.read_point_cloud(self.target_path)
            if self.init_target_path:
                self.init_target = o3d.io.read_point_cloud(self.init_target_path)
            return True
        except Exception as e:
            print(f"Error loading point clouds: {e}")
            return False

    def scale_point_cloud(self, point_cloud, scale_factor=1000):
        """Scales the given Open3D point cloud."""
        point_cloud.points = o3d.utility.Vector3dVector(np.asarray(point_cloud.points)/scale_factor)
        return point_cloud

    def preprocess_point_cloud(self, pcd, voxel_size):
        """Downsample and estimate normals for feature extraction."""
        pcd_down = pcd.voxel_down_sample(voxel_size)
        pcd_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=10))
        pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            pcd_down, o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=50)
        )
        return pcd_down, pcd_fpfh

    def pairwise_feature_matching(self, source_fpfh, target_fpfh):
        """Find nearest neighbor matches between two feature sets."""
        source_features = np.asarray(source_fpfh.data).T
        target_features = np.asarray(target_fpfh.data).T

        from scipy.spatial import cKDTree
        target_tree = cKDTree(target_features)
        distances, indices = target_tree.query(source_features, k=1)
        correspondences = np.vstack((np.arange(len(indices)), indices.flatten())).T
        return correspondences

    def execute_ransac(self, source_down, target_down, correspondences):
        """Perform transformation estimation using RANSAC."""
        distance_threshold = self.voxel_size * 1.5
        result = o3d.pipelines.registration.registration_ransac_based_on_correspondence(
            source_down, target_down, o3d.utility.Vector2iVector(correspondences),
            max_correspondence_distance=distance_threshold,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            ransac_n=5,
            criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999)
        )
        return result

    def refine_with_icp(self, source, target, initial_transform):
        """Refine alignment using ICP."""
        max_correspondence_distance = self.voxel_size * 0.4
        result_icp = o3d.pipelines.registration.registration_icp(
            source, target, max_correspondence_distance, initial_transform,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=1000)
        )
        return result_icp

    def remove_outliers(self, pcd, nb_neighbors=10, std_ratio=2.0, nb_points=5, radius=0.02):
        """Remove statistical and radius outliers."""
        cl, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
        inlier_cloud = pcd.select_by_index(ind)
        cl_2, ind_2 = inlier_cloud.remove_radius_outlier(nb_points=nb_points, radius=radius)
        return cl_2

    def register_point_clouds(self):
        if not self.load_point_clouds():
            return None
        scaled_source = self.scale_point_cloud(self.source)
        outlier_source = self.remove_outliers(scaled_source)
        outlier_target = self.remove_outliers(self.target)

        source_down, source_fpfh = self.preprocess_point_cloud(outlier_source, self.voxel_size_source)
        target_down, target_fpfh = self.preprocess_point_cloud(outlier_target, self.voxel_size_target)

        correspondences = self.pairwise_feature_matching(source_fpfh, target_fpfh)
        ransac_result = self.execute_ransac(source_down, target_down, correspondences)
        icp_result = self.refine_with_icp(outlier_source, outlier_target, ransac_result.transformation)

        return icp_result.transformation

    def visualize_transform(self, source, transformation):
        source_temp = source.transform(transformation)
        source_temp.paint_uniform_color([1, 0, 0])
        self.target.paint_uniform_color([0, 1, 1])
        if self.init_target is not None:
            o3d.visualization.draw_geometries([source_temp, self.target, self.init_target])
        else:
            o3d.visualization.draw_geometries([source_temp, self.target])

# if __name__ == "__main__":
#     source_path = "/home/airlab/Desktop/DigitalTwin_PoseEstimation/data/source/d435_on_ur10e_arm/source_ob1_0.ply"
#     target_path = '/home/airlab/Desktop/DigitalTwin_PoseEstimation/scripts/partial_point_clouds/view4.ply'
#     init_target_path = '/home/airlab/Desktop/DigitalTwin_PoseEstimation/data/ply_models/obj1.ply'

#     registrator = PointCloudRegistrator(source_path, target_path, init_target_path)
#     transformation = registrator.register_point_clouds()

#     if transformation is not None:
#         print("Transformation matrix:", transformation)
#         registrator.visualize_registration(transformation)