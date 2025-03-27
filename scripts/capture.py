import pyrealsense2 as rs
import numpy as np
import cv2
import torch
import open3d as o3d
# from depth_anything_v2.dpt import DepthAnythingV2
# from d435_capture import *
from global_registration import *
from PIL import Image

import pyrealsense2 as rs
import numpy as np
import cv2

import pyrealsense2 as rs
import numpy as np
import cv2
import open3d as o3d

# Stereo Matching Parameters
window_size = 5
min_disp = 0
num_disp = 96  # Must be divisible by 16

stereo = cv2.StereoSGBM_create(
    minDisparity=min_disp,
    numDisparities=num_disp,
    blockSize= 25,
    P1=8 * 3 * window_size**2,
    P2=32 * 3 * window_size**2,
    disp12MaxDiff=1,
    uniquenessRatio=12,
    speckleWindowSize=100,
    speckleRange= 32 
)

# Configure RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()

# Enable IR stereo streams and color stream
config.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 30)  # IR Left
config.enable_stream(rs.stream.infrared, 2, 640, 480, rs.format.y8, 30)  # IR Right
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)  # Color stream

# Start streaming
pipeline.start(config)

# Camera intrinsic parameters (RealSense D435 Example)
try:
    while True:
        # Capture frames
        frames = pipeline.wait_for_frames()
        ir_left = frames.get_infrared_frame(1)
        ir_right = frames.get_infrared_frame(2)
        color_frame = frames.get_color_frame()
     


        if not ir_left or not ir_right or not color_frame:
            continue

        # Convert frames to numpy arrays
        ir_left_img = np.asanyarray(ir_left.get_data())
        ir_right_img = np.asanyarray(ir_right.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Compute disparity map
        disparity = stereo.compute(ir_left_img, ir_right_img).astype(np.float32) / 16.0

        # Normalize disparity for visualization
        disp_vis = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        color_disparity = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)

        # Show images
        cv2.imshow('IR Left', ir_left_img)
        cv2.imshow('IR Right', ir_right_img)
        cv2.imshow("Disparity Map", color_disparity)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()

