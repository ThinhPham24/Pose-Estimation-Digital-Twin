# ----------------------------------------------------------------------------
# 20250307-QHD: Based on Thinh's solution
# https://github.com/ECERobot/Vision-Guided-robot-manipulation/blob/main/main_robot_operation.py
# https://github.com/ECERobot/Vision-Guided-robot-manipulation/blob/main/cvfun.py
# ----------------------------------------------------------------------------

import numpy as np
import pyrealsense2 as rs
import cv2
import open3d as o3d

def initialize (rs):

    # Initialize RealSense pipeline
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    pipeline.start(config)

    # Initialize RealSense alignment
    alignment = rs.align(rs.stream.color)

    return pipeline, alignment

def stop (pipeline):

    # Stop the RealSense pipeline when done
    pipeline.stop()

# Function to capture one frame at a time
def capture_frame(pipeline, align):
    # Capture a single set of frames
    frameset = pipeline.wait_for_frames()
    frameset = align.process(frameset)

    color_frame = frameset.get_color_frame()
    depth_frame = frameset.get_depth_frame()
    # # Adjust camera settings
    # device = pipeline.get_active_profile().get_device()
    # sensor = device.query_sensors()[1]  # Assuming color sensor is the second
    # sensor.set_option(rs.option.enable_auto_exposure, 1)  # 1 to enable, 0 to disable
    # # Enable auto exposure
    # sensor.set_option(rs.option.exposure, 500)  # Increase exposure
    # sensor.set_option(rs.option.gain, 16)  # Adjust gain
    # Get camera intrinsics
    # Get the intrinsics of the depth sensor
    depth_intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
    # Ensure frames are valid
    if not color_frame or not depth_frame:
        print("Skipping frame due to missing data.")
        return None, None
    
    return color_frame, depth_frame, depth_intrinsics

def get_point_cloud(color_image,depth_intrinsics, depth_image):
    height, width = depth_image.shape
    points = []
    colors = []
    for v in range(height):
        for u in range(width):
            # Get the depth value at pixel (u, v)
            depth = depth_image[v, u]
            if depth == 0:  # Skip invalid depth
                continue
            # Convert (u, v, depth) to 3D point in camera coordinates
            point = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [u, v], depth)
            # Append the 3D point and the corresponding color
            points.append(point)
            colors.append(color_image[v, u] / 255.0)  # Normalize color to [0, 1] for Open3D
            #colors.append(color_image[v, u])  # Normalize color to [0, 1] for Open3D
    # Convert points and colors to numpy arrays
    points = np.array(points)
    colors = np.array(colors)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def remove_flatground(pcd): 
    z_threshold = 100
    #z_threshold = 0.5
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    mask = points[:,2] > z_threshold
    pcd.points = o3d.utility.Vector3dVector(points[mask]) # normals and colors are unchanged
    pcd.colors = o3d.utility.Vector3dVector(colors[mask]) # normals and colors are unchanged

    return pcd

def pc_plane_seg_ransac(pcd):
    
    plane_model, inliers = pcd.segment_plane(distance_threshold=0.5,
                                         ransac_n=3,
                                            num_iterations=1000)

    inlier_cloud = pcd.select_by_index(inliers)
    inlier_cloud.paint_uniform_color([1.0, 0, 0])
    outlier_cloud = pcd.select_by_index(inliers, invert=True)

    return inlier_cloud, outlier_cloud
import os
import pyrealsense2 as rs
import cv2
import numpy as np
class stereo_realsense:
    def __init__(self):
        # self.pipeline = rs.pipeline()
        # config = rs.config()

        # # Enable IR and depth streams
        # config.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 30)
        # config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        # config.enable_stream(rs.stream.infrared, 2, 640, 480, rs.format.y8, 30)
        # config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

        # # Start streaming
        # self.profile = self.pipeline.start(config)
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 30)
        config.enable_stream(rs.stream.infrared, 2, 640, 480, rs.format.y8, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.profile = self.pipeline.start(config)
        # # Get the depth sensor and disable the emitter (dot projector)
        sensor = self.profile.get_device().first_depth_sensor()
        if sensor.supports(rs.option.emitter_enabled):
            sensor.set_option(rs.option.emitter_enabled, 0)  # 0 = OFF, 1 = ON
    def capture_frame(self):
        frames = self.pipeline.wait_for_frames()
        ir1 = frames.get_infrared_frame(1)
        ir2 = frames.get_infrared_frame(2)
        ir1_img = np.asanyarray(ir1.get_data())
        ir2_img = np.asanyarray(ir2.get_data())
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        depth_intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
        if not color_frame or not depth_frame:
            print("Skipping frame due to missing data.")
            return None, None
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        return color_image, ir1_img, ir2_img, depth_image, depth_intrinsics
    def release(self):
        self.pipeline.stop()



        

   