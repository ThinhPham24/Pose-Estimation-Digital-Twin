import pyrealsense2 as rs
import numpy as np
import cv2
import torch
import open3d as o3d
from depth_anything_v2.dpt import DepthAnythingV2
from utils.d435_capture import *
from utils.global_registration import *
from PIL import Image
from ultralytics import YOLOE
import numpy as np
from ultralytics.models.yolo.yoloe.predict_vp import YOLOEVPSegPredictor
from ultralytics.engine.results import Results

ply_header = '''ply
format ascii 1.0
element vertex %(vert_num)d
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
'''
def write_ply(fn, verts, colors):
    # verts = verts.reshape(-1, 3)
    # colors = colors.reshape(-1, 3)
    verts = np.hstack([verts, colors])
    with open(fn, 'wb') as f:
        f.write((ply_header % dict(vert_num=len(verts))).encode('utf-8'))
        np.savetxt(f, verts, fmt='%f %f %f %d %d %d ')
def write_ply_nColor(fn, verts, colors):
    verts = verts.reshape(-1, 3)
    colors = colors.reshape(-1, 3)
    verts = np.hstack([verts, colors])
    with open(fn, 'wb') as f:
        f.write((ply_header % dict(vert_num=len(verts))).encode('utf-8'))
        np.savetxt(f, verts, fmt='%f %f %f %d %d %d ')
if __name__ == "__main__":
    # DEVICE = "cuda: 1" if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    gpu_id = 0  # Change this based on the available GPUs
    if torch.cuda.device_count() > gpu_id:
        DEVICE = torch.device(f"cuda:{gpu_id}")
    else:
        DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    # Depth Any Thing Model
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }

    encoder = 'vitl'  # Select encoder: 'vits', 'vitb', 'vitl', 'vitg'
    max_depth = 1
    depth_any_thing = DepthAnythingV2(**model_configs[encoder])
    depth_any_thing.load_state_dict(torch.load(R'/home/airlab/Desktop/DigitalTwin_PoseEstimation/scripts/depth_anything_v2/checkpoints/depth_anything_v2_vitl.pth', map_location=DEVICE))
    depth_any_thing = depth_any_thing.to(DEVICE).eval()
    # Initialize RealSense
    # pipeline, alignment = initialize(rs)
    # i = 0
    # while True:
    #     color_frame, depth_frame, depth_intrinsics = capture_frame(pipeline, alignment)

    #     if color_frame is None or depth_frame is None:
    #         print("Warning: No frames captured")
    #         continue

    #     color_image = np.asanyarray(color_frame.get_data())
    #     depth_image = np.asanyarray(depth_frame.get_data())
    image = cv2.imread("/home/airlab/Desktop/DigitalTwin_PoseEstimation/data/images/Test3.png")
    # re_image = cv2.resize(image, (640,640), cv2.INTER_AREA)
    image_color = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Read the image using OpenCV
    pred_depth = depth_any_thing.infer_image(image_color)  # Depth estimation model output
    # Resize depth prediction to match original image size
    # resized_pred_depth = np.array(Image.fromarray(pred_depth).resize((640, 640), Image.NEAREST))
    # Normalize disparity for depth calculation
    disparity = pred_depth.astype(np.float32) / 16  # Adjust divisor based on stereo model output
    pred_depth[pred_depth == 0] = 1e-6  # Prevent division by zero
    print("Depth", pred_depth.shape )
    # Normalize depth for visualization
    depth_visual = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    # depth_camera = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    color_applied = cv2.applyColorMap(depth_visual,cv2.COLORMAP_JET)
    # Show depth map
    cv2.imshow("Depth map", color_applied)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # Generate 3D point cloud
    # cx, cy = 329.27, 244.46  # Principal point
    cx = 320
    cy = 320
    x, y = np.meshgrid(np.arange(640), np.arange(480))
    # fx = 615.75  #Focal length (pixels)
    # fy = 616.02
    fx = 470.4
    fy = 470.4
    x = ((x - cx) / fx) # model is trained by local lenght = 470.4
    y = ((y - cy) / fy)
    # Convert depth to float and scale it to meters
    z = pred_depth/1000 # 
    # Stack into (X, Y, Z) coordinates
    points = np.stack((np.multiply(x, z), np.multiply(y, z), z), axis=-1).reshape(-1, 3)
    # Normalize color data
    colors = np.array(image_color).reshape(-1, 3) / 255.0
    # Create Open3D Point Cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)  
    mask = points[:,2] > points[:,2].min()
    pcd.points = o3d.utility.Vector3dVector(points[mask]) # normals and colors are unchanged                             
    pcd.colors = o3d.utility.Vector3dVector(colors[mask]) # normals and colors are unchanged
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=10))
    pcd.orient_normals_consistent_tangent_plane(k=5) 
    i = 1
    o3d.io.write_point_cloud(R"/home/airlab/Desktop/DigitalTwin_PoseEstimation/scripts/source_ob1_{}.ply".format(i), pcd, write_ascii=True)
    # i += 1
    # write_ply('point_cloud.ply',points[mask], colors[mask])
    # Visualize
    o3d.visualization.draw_geometries([pcd]) 
