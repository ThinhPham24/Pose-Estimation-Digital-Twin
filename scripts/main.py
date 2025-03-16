import pyrealsense2 as rs

from d435_capture import *
from global_registration import *

if __name__ == "__main__":

    try:
        
        # ----------------------------------------------------------------------------------
        # 1. Capture PC (Scene)
        # ----------------------------------------------------------------------------------

        # pipeline, alignment = initialize(rs)

        # color_frame, depth_frame, depth_intrinsics = capture_frame (pipeline, alignment)

        # # Convert depth frame to a numpy array
        # color_image = np.asanyarray(color_frame.get_data())
        # depth_image = np.asanyarray(depth_frame.get_data())

        # point_cloud = get_point_cloud(color_image, depth_intrinsics, depth_image)

        # o3d.visualization.draw_geometries([ point_cloud])

        # o3d.io.write_point_cloud("../data/source/raw_data_from_d435_on_ur10e_arm.ply", point_cloud, write_ascii=True)

        # ----------------------------------------------------------------------------------
        # 2. Segment RANSAC & Remove flatground
        # ----------------------------------------------------------------------------------

        # Do not know why relative path does not work yet. Have to use full path
        #pcd = o3d.io.read_point_cloud("../data/source/d435_on_ur10e_arm/raw.ply")
        #pcd = o3d.io.read_point_cloud("/home/ece/new_ws/DigitalTwin_PoseEstimation/data/source/d435_on_ur10e_arm/raw.ply")

        # removed_flatground_pcd = remove_flatground (pcd)
        # o3d.visualization.draw_geometries([removed_flatground_pcd])

        #inlier_cloud, outlier_cloud = pc_plane_seg_ransac (pcd)
        #o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])

        #o3d.io.write_point_cloud("/home/ece/new_ws/DigitalTwin_PoseEstimation/data/source/d435_on_ur10e_arm/removed_flatground.ply", removed_flatground_pcd, write_ascii=True)
        
        # ----------------------------------------------------------------------------------
        # 3. Remove outliers (Kdtree (4))
        # ----------------------------------------------------------------------------------

        # Manually refined the source

        # ----------------------------------------------------------------------------------
        # 4. Registration methods
        # ----------------------------------------------------------------------------------        
        source = o3d.io.read_point_cloud(R"C:\Users\Thinh\Desktop\DigitalTwin_PoseEstimation\data\source\d435_on_ur10e_arm\Source_ob1.ply")
        target = o3d.io.read_point_cloud(R"C:\Users\Thinh\Desktop\DigitalTwin_PoseEstimation\data\ply_models\obj1.ply")

        # # Scale from mm to m
        source = scale_point_cloud (source)

        # ----------------------------------------------------------------------------------
        # 4.1. Global registration - RANSAC
        # ----------------------------------------------------------------------------------        
        voxel_size = 0.01
        source_voxel_size = 0.005
        target_voxel_size = 0.005

        source_down, source_fpfh = preprocess_point_cloud_source(source, source_voxel_size)

        target_down, target_fpfh = preprocess_point_cloud(
                                        target, target_voxel_size)
        
        o3d.visualization.draw_geometries([source_down])

        result_ransac = execute_global_ransac_registration(source_down, target_down,
                                                    source_fpfh, target_fpfh,
                                                    voxel_size)

        if (result_ransac.transformation.trace() == 4.0):
            success = False
        else:
            success = True

        draw_registration_result(source_down, target_down, result_ransac.transformation)

        print("RANSAC complete. Refining with ICP...")
        max_correspondence_distance = 0.002  # 2 cm refinement for ICP
        result_icp = o3d.pipelines.registration.registration_icp(
            source, target, max_correspondence_distance, result_ransac.transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000)
        )

        print("ICP refinement complete.")

        draw_registration_result(source, target, result_icp.transformation)




    finally:
        # Stop the RealSense pipeline when done
        # stop(pipeline)
        cv2.destroyAllWindows()