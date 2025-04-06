
# import os
# DEVICE = 'cuda'

# # def load_image(imfile):
# #     img = np.array(Image.open(imfile)).astype(np.uint8)
# #     img = torch.from_numpy(img).permute(2, 0, 1).float()
# #     return img[None].to(DEVICE)
# # Create a pipeline and config
# pipeline = rs.pipeline()
# config = rs.config()

# # Enable IR and depth streams
# config.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 30)
# config.enable_stream(rs.stream.infrared, 2, 640, 480, rs.format.y8, 30)
# config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# # Start streaming
# profile = pipeline.start(config)

# # Get the depth sensor and disable the emitter (dot projector)
# sensor = profile.get_device().first_depth_sensor()
# if sensor.supports(rs.option.emitter_enabled):
#     sensor.set_option(rs.option.emitter_enabled, 0)  # 0 = OFF, 1 = ON
#     print("Emitter turned OFF")

# # Continue with your streaming loop
# output_dir = "/home/airlab/Desktop/DEFOM-Stereo/"

# i = 0
# try:
#     while True:
       
#         frames = pipeline.wait_for_frames()
#         ir1 = frames.get_infrared_frame(1)
#         ir2 = frames.get_infrared_frame(2)

#         ir1_img = np.asanyarray(ir1.get_data())
#         ir2_img = np.asanyarray(ir2.get_data())

#         cv2.imshow("IR Left (Emitter Off)", ir1_img)
#         cv2.imshow("IR Right (Emitter Off)", ir2_img)
        
#         if cv2.waitKey(1) & 0xFF == ord('s'):
#             print("Save")
#             save_path_l = os.path.join(f"{output_dir}/Stereo_images", f'L_{i}.png')
#             cv2.imwrite(save_path_l, ir1_img)
#             save_path_r = os.path.join(f"{output_dir}/Stereo_images", f'R_{i}.png')
#             cv2.imwrite(save_path_r, ir2_img)
#             i += 1
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
# finally:
#     pipeline.stop()
#     cv2.destroyAllWindows()
import pyrealsense2 as rs
import cv2
import sys
sys.path.append('core')

import argparse
import glob
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path
from core.defom_stereo import DEFOMStereo
from core.utils.utils import InputPadder
from PIL import Image
from matplotlib import pyplot as plt
import open3d as o3d


def visualize_point_cloud(points, colors):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd])
    return pcd
   
DEVICE = 'cuda'

def load_image(imfile):
    img = Image.open(imfile).convert("RGB")
    img = img.resize((640, 480), resample=Image.BILINEAR)
    img = np.array(img).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)

# Init setup model
parser = argparse.ArgumentParser()
parser.add_argument('--restore_ckpt',type=str, default="/home/airlab/Desktop/DigitalTwin_PoseEstimation/scripts/checkpoints/defomstereo_vitl_sceneflow.pth", help="restore checkpoint")
parser.add_argument('--save_numpy', action='store_true', help='save output as numpy arrays')
parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
parser.add_argument('--valid_iters', type=int, default=32, help='number of flow-field updates during forward pass')
parser.add_argument('--scale_iters', type=int, default=8, help="number of scaling updates to the disparity field in each forward pass.")

# Architecture choices
parser.add_argument('--dinov2_encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl', 'vitg'])
parser.add_argument('--idepth_scale', type=float, default=0.5, help="the scale of inverse depth to initialize disparity")
parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128]*3, help="hidden state and context dimensions")
parser.add_argument('--corr_implementation', choices=["reg", "alt", "reg_cuda", "alt_cuda"], default="reg", help="correlation volume implementation")
parser.add_argument('--shared_backbone', action='store_true', help="use a single backbone for the context and feature encoders")
parser.add_argument('--corr_levels', type=int, default=2, help="number of levels in the correlation pyramid")
parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")
parser.add_argument('--scale_list', type=float, nargs='+', default=[0.125, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0],
                    help='the list of scaling factors of disparity')
parser.add_argument('--scale_corr_radius', type=int, default=2,
                    help="width of the correlation pyramid for scaled disparity")

parser.add_argument('--n_downsample', type=int, default=2, choices=[2, 3], help="resolution of the disparity field (1/2^K)")
parser.add_argument('--context_norm', type=str, default="instance", choices=['group', 'batch', 'instance', 'none'], help="normalization of context encoder")
parser.add_argument('--n_gru_layers', type=int, default=3, help="number of hidden GRU levels")
args = parser.parse_args()

if __name__ =="__main__":
    model = DEFOMStereo(args)
    checkpoint = torch.load(args.restore_ckpt, map_location='cuda')

    if 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'],strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)

    model.to(DEVICE)
    model.eval()
    left_imgs = '/home/airlab/Desktop/DigitalTwin_PoseEstimation/scripts/Stereo_images/*_left.png'
    right_imgs ='/home/airlab/Desktop/DigitalTwin_PoseEstimation/scripts/Stereo_images/*_right.png'
    output_directory = '/home/airlab/Desktop/DigitalTwin_PoseEstimation/scripts/Stereo_images/'
    output_directory = Path(output_directory)
    output_directory.mkdir(exist_ok=True)

    with torch.no_grad():
        left_images = sorted(glob.glob(left_imgs, recursive=True))
        right_images = sorted(glob.glob(right_imgs, recursive=True))
        print(f"Found {len(left_images)} images. Saving files to {output_directory}/")

        for (imfile1, imfile2) in tqdm(list(zip(left_images, right_images))):
            image1 = load_image(imfile1)
            image2 = load_image(imfile2)

            padder = InputPadder(image1.shape, divis_by=32)
            image1, image2 = padder.pad(image1, image2)

            with torch.no_grad():
                disp_pr = model(image1, image2, iters=args.valid_iters, scale_iters=args.scale_iters, test_mode=True)
            disp_pr = padder.unpad(disp_pr).cpu().squeeze().numpy()
            # Calibration
            fx, fy, cx1, cy = 615.75, 616.02, 329.27, 244.46
            cx2 = 329.27
            baseline= 50 # in millimeters #50mm
            # inverse-project
            depth = ((fx * baseline) / (-disp_pr + (cx2 - cx1)))
            print("Depth", depth)
            H, W = depth.shape
            xx, yy = np.meshgrid(np.arange(W), np.arange(H))
            points_grid = np.stack(((xx-cx1)/fx, (yy-cy)/fy, np.ones_like(xx)), axis=0) * depth

            mask = np.ones((H, W), dtype=bool)

            # Remove flying points
            mask[1:][np.abs(depth[1:] - depth[:-1]) > 1] = False
            mask[:,1:][np.abs(depth[:,1:] - depth[:,:-1]) > 1] = False
            image = cv2.imread(imfile1)
            points = points_grid.transpose(1,2,0)[mask]
            colors = image[mask].astype(np.float64) / 255
            pcd = visualize_point_cloud(points, colors)
            file_stem = imfile1.split('/')[-1].split('_')[0]+'_'+args.restore_ckpt.split('/')[-1][:-4]
            # if args.save_numpy:
            o3d.io.write_point_cloud(output_directory / "scene.ply", pcd, write_ascii=True)
            # file_stem = imfile1.split('/')[-1].split('_')[0]+'_'+args.restore_ckpt.split('/')[-1][:-4]
            # if args.save_numpy:
            #     np.save(output_directory / f"{file_stem}.npy", disp_pr)
            # plt.imsave(output_directory / f"{file_stem}.png", disp_pr, cmap='jet')
