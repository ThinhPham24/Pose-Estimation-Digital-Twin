import pyrealsense2 as rs
import numpy as np
import cv2
import torch
import open3d as o3d
from depth_anything_v2.dpt import DepthAnythingV2
from d435_capture import *
from global_registration import *
from PIL import Image
if __name__ == "__main__":

    # Initialize RealSense
    pipeline, alignment = initialize(rs)

    while True:
        color_frame, depth_frame, depth_intrinsics = capture_frame(pipeline, alignment)

        if color_frame is None or depth_frame is None:
            print("Warning: No frames captured")
            continue

        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        h, w = color_image.shape[:2]
        cv2.imshow("color", color_image)
        k = cv2.waitKey(0)
        if k == ord('s'):
            cv2.imwrite("color.png", color_image)