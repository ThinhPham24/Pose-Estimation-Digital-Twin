import pyrealsense2 as rs
import numpy as np
import cv2

try:
    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30) #color stream

    # Start streaming
    pipeline.start(config)

    # Wait for a coherent pair of frames: depth and color
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    if not color_frame:
        raise Exception("Could not get color frame")

    # Convert images to numpy arrays
    color_image = np.asanyarray(color_frame.get_data())

    # Display color image
    cv2.imshow('RealSense Color', color_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Optionally, save the image
    cv2.imwrite('realsense_color_image.png', color_image)

    # Stop streaming
    pipeline.stop()

except Exception as e:
    print(e)
