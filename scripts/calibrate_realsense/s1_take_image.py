import os
import pyrealsense2 as rs
import cv2
import numpy as np
pipeline = rs.pipeline()
config = rs.config()

# Enable IR and depth streams
config.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 30)
config.enable_stream(rs.stream.infrared, 2, 640, 480, rs.format.y8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# Start streaming
profile = pipeline.start(config)

# Get the depth sensor and disable the emitter (dot projector)
sensor = profile.get_device().first_depth_sensor()
if sensor.supports(rs.option.emitter_enabled):
    sensor.set_option(rs.option.emitter_enabled, 0)  # 0 = OFF, 1 = ON
    print("Emitter turned OFF")

# Continue with your streaming loop
output_dir = "/home/airlab/Desktop/DEFOM_Stereo/"

i = 0
try:
    while True:
       
        frames = pipeline.wait_for_frames()
        ir1 = frames.get_infrared_frame(1)
        ir2 = frames.get_infrared_frame(2)

        ir1_img = np.asanyarray(ir1.get_data())
        ir2_img = np.asanyarray(ir2.get_data())

        cv2.imshow("IR Left (Emitter Off)", ir1_img)
        cv2.imshow("IR Right (Emitter Off)", ir2_img)
        
        if cv2.waitKey(1) & 0xFF == ord('s'):
            print("Save", i)
            save_path_l = os.path.join(f"{output_dir}/Images", f'L_{i}.png')
            cv2.imwrite(save_path_l, ir1_img)
            save_path_r = os.path.join(f"{output_dir}/Images", f'R_{i}.png')
            cv2.imwrite(save_path_r, ir2_img)
            i += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    pipeline.stop()
    cv2.destroyAllWindows()