import numpy as np
import open3d as o3d
from scipy.spatial import ConvexHull
import math
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
    def draw_pca_bounding_box(self, pcd, color=[0, 1, 0]):
        # Convert point cloud to numpy array
        points = np.asarray(pcd.points)

        # Compute centroid
        centroid = np.mean(points, axis=0)

        # Subtract centroid
        centered_points = points - centroid

        # PCA via SVD
        U, S, Vt = np.linalg.svd(centered_points, full_matrices=False)
        R = Vt.T  # Rotation matrix (principal axes)

        # Transform points into PCA frame
        aligned = centered_points @ R

        # Compute min and max along each PCA axis
        min_bound = np.min(aligned, axis=0)
        max_bound = np.max(aligned, axis=0)

        # Compute 8 corner points in the aligned frame
        bbox_corners = np.array([
            [min_bound[0], min_bound[1], min_bound[2]],
            [max_bound[0], min_bound[1], min_bound[2]],
            [max_bound[0], max_bound[1], min_bound[2]],
            [min_bound[0], max_bound[1], min_bound[2]],
            [min_bound[0], min_bound[1], max_bound[2]],
            [max_bound[0], min_bound[1], max_bound[2]],
            [max_bound[0], max_bound[1], max_bound[2]],
            [min_bound[0], max_bound[1], max_bound[2]],
        ])

        # Transform corners back to original coordinate system
        bbox_corners_world = (bbox_corners @ R.T) + centroid

        # Create bounding box mesh
        lines = [
            [0,1],[1,2],[2,3],[3,0],
            [4,5],[5,6],[6,7],[7,4],
            [0,4],[1,5],[2,6],[3,7]
        ]
        colors = [color for _ in range(len(lines))]

        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(bbox_corners_world),
            lines=o3d.utility.Vector2iVector(lines)
        )
        line_set.colors = o3d.utility.Vector3dVector(colors)

        return line_set
    def visualize_transform(self, source, transformation):
        source_temp = source.transform(transformation)
        source_temp.paint_uniform_color([0.5, 0.5, 0])
        self.target.paint_uniform_color([0, 0.7, 0.25])
        # if self.init_target is not None:
        #     o3d.visualization.draw_geometries([source_temp, self.target, self.init_target])
        # else:
        #     o3d.visualization.draw_geometries([source_temp, self.target])
         # Create a coordinate frame and apply the same transformation
        center = source_temp.get_center()
        # # Create a transformation matrix to move the coordinate frame to the center of the bounding box
        T = np.eye(4)
        T[:3, 3] =  center
        theta = np.deg2rad(90)  # Convert 90 degrees to radians
        Rz_90 = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta),  np.cos(theta), 0],
            [0,              0,             1]
        ])
        zita = np.deg2rad(180)  # Convert 180 degrees to radians
        Ry_180 = np.array([
            [ np.cos(zita), 0, np.sin(zita)],
            [ 0,             1, 0            ],
            [-np.sin(zita), 0, np.cos(zita)]
        ])
        rot_temp = transformation[:3, :3] @ Rz_90 @ Ry_180
        T[:3, :3] = rot_temp
        # print("Transformation T", T)
        # Transform to camera frame: X_cam = R @ X_obj + t
        axis_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
        axis_frame.transform(T)
        # bbox =self.draw_pca_bounding_box(source_temp)

        # Prepare geometries
        geometries = [source_temp, self.target, axis_frame]
        if self.init_target is not None:
            geometries.append(self.init_target)

        # Visualize
        o3d.visualization.draw_geometries(geometries)
        # o3d.visualization.draw_geometries_with_editing(geometries)
        o3d.io.write_point_cloud("/home/airlab/Desktop/DigitalTwin_PoseEstimation/Pose_results/source_temp.ply", source_temp, write_ascii=True)
        o3d.io.write_point_cloud("/home/airlab/Desktop/DigitalTwin_PoseEstimation/Pose_results/target.ply", self.target, write_ascii=True)
        o3d.io.write_triangle_mesh("/home/airlab/Desktop/DigitalTwin_PoseEstimation/Pose_results/axis_frame.ply", axis_frame, write_ascii=True)

# if __name__ == "__main__":
#     source_path = "/home/airlab/Desktop/DigitalTwin_PoseEstimation/data/source/d435_on_ur10e_arm/source_ob1_0.ply"
#     target_path = '/home/airlab/Desktop/DigitalTwin_PoseEstimation/scripts/partial_point_clouds/view4.ply'
#     init_target_path = '/home/airlab/Desktop/DigitalTwin_PoseEstimation/data/ply_models/obj1.ply'

#     registrator = PointCloudRegistrator(source_path, target_path, init_target_path)
#     transformation = registrator.register_point_clouds()

#     if transformation is not None:
#         print("Transformation matrix:", transformation)
#         registrator.visualize_registration(transformation)