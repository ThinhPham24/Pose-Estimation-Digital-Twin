import pyrealsense2 as rs
import numpy as np
import open3d as o3d

# Configure depth & color streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start the pipeline
pipeline.start(config)

# Align depth to color
align_to = rs.stream.color
align = rs.align(align_to)

# Initialize depth filters
decimation = rs.decimation_filter(magnitude=2)
spatial = rs.spatial_filter()
spatial.set_option(rs.option.filter_smooth_alpha, 0.5)
spatial.set_option(rs.option.filter_smooth_delta, 20)
spatial.set_option(rs.option.filter_magnitude, 2)
spatial.set_option(rs.option.holes_fill, 1)

temporal = rs.temporal_filter()
hole_filling = rs.hole_filling_filter()

try:
    for _ in range(10):  # Capture multiple frames for better averaging
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue

        # Apply depth filters
        depth_frame = decimation.process(depth_frame)
        depth_frame = spatial.process(depth_frame)
        depth_frame = temporal.process(depth_frame)
        depth_frame = hole_filling.process(depth_frame)

        # Convert to numpy
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Generate point cloud
        pc = rs.pointcloud()
        points = pc.calculate(depth_frame)
        pc.map_to(color_frame)

        # Convert to Open3D format
        vtx = np.asanyarray(points.get_vertices())  # XYZ points
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(vtx.view(np.float32).reshape(-1, 3))
        pcd.colors = o3d.utility.Vector3dVector(color_image.reshape(-1, 3) / 255.0)

        # Save high-accuracy point cloud
        # o3d.io.write_point_cloud("high_accuracy_pcd.ply", pcd)
        o3d.visualization.draw_geometries([pcd])

        print("Saved high-accuracy point cloud: high_accuracy_pcd.ply")

finally:
    pipeline.stop()
