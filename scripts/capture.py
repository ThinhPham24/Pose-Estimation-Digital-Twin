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

    # # Initialize RealSense
    # pipeline, alignment = initialize(rs)

    # while True:
    #     color_frame, depth_frame, depth_intrinsics = capture_frame(pipeline, alignment)

    #     if color_frame is None or depth_frame is None:
    #         print("Warning: No frames captured")
    #         continue

    #     color_image = np.asanyarray(color_frame.get_data())
    #     depth_image = np.asanyarray(depth_frame.get_data())
    #     h, w = color_image.shape[:2]
    #     cv2.imshow("color", color_image)
    #     k = cv2.waitKey(0)
    #     if k == ord('s'):
    #         cv2.imwrite("Obj1.png", color_image)

    # Configure RealSense pipeline
    pipeline = rs.pipeline()
    config = rs.config()

    # Enable infrared streams (IR left = 1, IR right = 2)
    config.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 30)  # IR Left
    config.enable_stream(rs.stream.infrared, 2, 640, 480, rs.format.y8, 30)  # IR Right

    # Start pipeline
    pipeline.start(config)

    try:
        while True:
            # Wait for frames
            frames = pipeline.wait_for_frames()
            
            # Get infrared images
            ir_left = frames.get_infrared_frame(1)   # IR Left (Index 1)
            ir_right = frames.get_infrared_frame(2)  # IR Right (Index 2)

            if not ir_left or not ir_right:
                continue

            # Convert frames to numpy arrays
            ir_left_img = np.asanyarray(ir_left.get_data())
            ir_right_img = np.asanyarray(ir_right.get_data())

            # Display images
            cv2.imshow('IR Left', ir_left_img)
            cv2.imshow('IR Right', ir_right_img)

            # Save images (Optional)
            cv2.imwrite('ir_left.png', ir_left_img)
            cv2.imwrite('ir_right.png', ir_right_img)

            # Press 'q' to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        # Stop the pipeline
        pipeline.stop()
        cv2.destroyAllWindows()
