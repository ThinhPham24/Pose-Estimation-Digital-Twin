import pyrealsense2 as rs
import cv2
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
from ultralytics import YOLOE
from ultralytics.models.yolo.yoloe.predict_vp import YOLOEVPSegPredictor
from ultralytics.engine.results import Results
from ultralytics.engine.results import Boxes
from utils.matching_3D import PointCloudRegistrator
import trimesh
import os
from utils.draw_bbox import draw_detections, draw_3d_bbox_on_point_cloud
import time
class ObjectPointCloudGenerator:
    def __init__(self, defom_ckpt, yoloe_ckpt, device="cuda:0"):
        self.device = torch.device(device) if torch.cuda.is_available() else torch.device("cpu")
        self.defom_stereo_model = self._load_defom_model(defom_ckpt)
        self.see_any_thing = self._load_yoloe(yoloe_ckpt)
        # self.cv_file = self._load_cailb_matrix(map_calib)

    def _load_defom_model(self, restore_ckpt):
        parser = argparse.ArgumentParser()
        parser.add_argument('--restore_ckpt', type=str, default=restore_ckpt, help="restore checkpoint")
        parser.add_argument('--save_numpy', action='store_true', help='save output as numpy arrays')
        parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
        parser.add_argument('--valid_iters', type=int, default=32, help='number of flow-field updates during forward pass')
        parser.add_argument('--scale_iters', type=int, default=8, help="number of scaling updates to the disparity field in each forward pass.")
        parser.add_argument('--dinov2_encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl', 'vitg'])
        parser.add_argument('--idepth_scale', type=float, default=0.5, help="the scale of inverse depth to initialize disparity")
        parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128] * 3, help="hidden state and context dimensions")
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
        model = DEFOMStereo(args)
        model.load_state_dict(torch.load(args.restore_ckpt, map_location=self.device), strict=False)
        model.to(self.device)
        model.eval()
        return model

    def _load_yoloe(self, model_path, source_image='ultralytics/assets/Test3.png', class_name=["cude"]):
        see_any_thing = YOLOE(model_path).to(self.device)
        visuals = dict(
            bboxes=np.array([[178.2, 218.1, 365.2, 372.1]]),
            cls=np.array([0])
        )
        image_size = 640
        see_any_thing.predict(source_image, imgsz=image_size, prompts=visuals, predictor=YOLOEVPSegPredictor, return_vpe=True)
        see_any_thing.set_classes(class_name, see_any_thing.predictor.vpe)
        see_any_thing.predictor = None
        return see_any_thing

    def apply_nms(self, results, iou_threshold=0.5):
        boxes = results.boxes.xyxy.cpu().numpy()
        scores = results.boxes.conf.cpu().numpy()
        masks = results.masks.data.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy()

        if len(boxes) == 0:
            return results

        selected_indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), score_threshold=0.1, nms_threshold=iou_threshold)

        if len(selected_indices) > 0:
            selected_indices = selected_indices.flatten().tolist()
            filtered_boxes = boxes[selected_indices]
            filtered_scores = scores[selected_indices]
            filtered_masks = masks[selected_indices]
            filtered_classes = classes[selected_indices]

            filtered_boxes_tensor = torch.tensor(filtered_boxes, device=results.boxes.xyxy.device)
            filtered_scores_tensor = torch.tensor(filtered_scores, device=results.boxes.conf.device)
            filtered_classes_tensor = torch.tensor(filtered_classes, device=results.boxes.cls.device)

            filtered_boxes_data = torch.cat([filtered_boxes_tensor, filtered_scores_tensor.unsqueeze(1), filtered_classes_tensor.unsqueeze(1)], dim=1)
            filtered_boxes_obj = Boxes(filtered_boxes_data, orig_shape=results.boxes.orig_shape)

            results.boxes = filtered_boxes_obj
            results.masks.data = torch.tensor(filtered_masks, device=results.masks.data.device)
        else:
            empty_boxes_data = torch.tensor([], device=results.boxes.xyxy.device).reshape(0, 6)
            empty_boxes = Boxes(empty_boxes_data, orig_shape=results.boxes.orig_shape)
            results.boxes = empty_boxes
            results.masks.data = torch.tensor([], device=results.masks.data.device).reshape(0, results.masks.data.shape[1], results.masks.data.shape[2], results.masks.data.shape[3])

        return results

    def compute_normal(self, points, colors):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=10))
        pcd.orient_normals_consistent_tangent_plane(k=5)
        return pcd

    def draw_contours_from_mask(self, image, mask, contour_color=(0, 255, 0), contour_thickness=1):
        if len(mask.shape) > 2:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        if mask.dtype != np.uint8:
            mask = mask.astype(np.uint8)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        if contours:
            max_cont = max(contours, key=cv2.contourArea)
            cv2.drawContours(image, [max_cont], -1, contour_color, contour_thickness)

        return image

    def draw_scaled_bbox(self, image, bbox_640x640, bbox_color=(255, 0, 0), bbox_thickness=2):
        bbox_640x480 = [
            int(bbox_640x640[0]),
            int(bbox_640x640[1]),
            int(bbox_640x640[2]),
            int(bbox_640x640[3]),
        ]

        cv2.rectangle(
            image,
            (bbox_640x480[0], bbox_640x480[1]),
            (bbox_640x480[2], bbox_640x480[3]),
            bbox_color,
            bbox_thickness,
        )

        return image

    def get_colored_point_cloud_from_bbox(self, depth_image, color_image, intrinsics, bbox):
        x_min, y_min, x_max, y_max = bbox
        points_3d = []
        colors_rgb = []

        for y in range(int(y_min), int(y_max)):
            for x in range(int(x_min), int(x_max)):
                depth = depth_image[y, x]
                if depth > 0:
                    point_3d = rs.rs2_deproject_pixel_to_point(intrinsics, [x, y], depth)
                    points_3d.append(np.array(point_3d) / 1000)
                    colors_rgb.append(color_image[y, x] / 255.0)

        return np.array(points_3d), np.array(colors_rgb)

    def get_colored_point_cloud_from_mask(self, depth_image, rgb_image, intrinsics, mask):
        """
        Inputs:
            mask: (H, W) binary mask of the object (0 or 1)
            depth: (H, W) depth map in meters
            rgb_image: (H, W, 3) RGB image aligned with depth
            intrinsics: (3, 3) camera intrinsics

        Returns:
            points_3d: (N, 3) 3D coordinates
            colors: (N, 3) RGB colors [0-255]
        """
        if isinstance(intrinsics, np.ndarray):
            fx, fy = intrinsics[0, 0], intrinsics[1, 1]
            cx, cy = intrinsics[0, 2], intrinsics[1, 2]
        else:
            fx, fy = intrinsics.fx, intrinsics.fy
            cx, cy = intrinsics.ppx, intrinsics.ppy

        # Find valid mask pixels
        ys, xs = np.where(mask > 0)
        zs = depth_image[ys, xs] / 1000

        valid = zs > 0
        xs, ys, zs = xs[valid], ys[valid], zs[valid]

        # Back-project to 3D
        x3d = (xs - cx) * zs / fx
        y3d = (ys - cy) * zs / fy
        z3d = zs
        points_3d = np.vstack((x3d, y3d, z3d)).T  # (N, 3)

        # Sample corresponding colors
        colors = rgb_image[ys, xs]  # (N, 3)

        return np.array(points_3d), np.array(colors)

    def visualize_point_cloud(self, points, colors):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        o3d.visualization.draw_geometries([pcd])
        return pcd

    def _mask_processing(self, mask):
        h, w = mask.shape[:2]
        binary_image = np.zeros((h, w), dtype=np.uint8)
        m = (mask * 255).astype(np.uint8)
        m = np.squeeze(m)
        contours, _ = cv2.findContours(m, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        mCnt = max(contours, key=cv2.contourArea)
        _mask = cv2.drawContours(binary_image, [mCnt], -1, 255, cv2.FILLED)
        kernel = np.ones((3, 3), np.uint8)
        post_mask = cv2.erode(_mask, kernel, iterations=1)
        post_mask = cv2.dilate(_mask, kernel, iterations=1)
        return post_mask

    def visualize(self, rgb, pred_rots, pred_trans, model_points, K, save_path):
        img = draw_detections(rgb, pred_rots, pred_trans, model_points, K, color=(255, 0, 0))
        img = Image.fromarray(np.uint8(img))
        img.save(save_path)
    def _load_cailb_matrix(self, map_calib_file):
        cv_file = cv2.FileStorage()
        cv_file.open(map_calib_file, cv2.FileStorage_READ)
        return cv_file

    def undistortRectify(self, frame, map_calib_file, left_true=True):
        cv_file = cv2.FileStorage()
        cv_file.open(map_calib_file, cv2.FileStorage_READ)
        stereoMapL_x = cv_file.getNode('stereoMapL_x').mat()
        stereoMapL_y = cv_file.getNode('stereoMapL_y').mat()
        stereoMapR_x = cv_file.getNode('stereoMapR_x').mat()
        stereoMapR_y = cv_file.getNode('stereoMapR_y').mat()
        if left_true:
            undistorted = cv2.remap(frame, stereoMapL_x, stereoMapL_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
        else:
            undistorted = cv2.remap(frame, stereoMapR_x, stereoMapR_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)

        return undistorted

    def load_image(self, image, map_calib_file, left_True=True):
        # img = Image.open(imfile).convert("RGB")
        # img = img.resize((640, 480), resample=Image.BILINEAR)
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (640,480), cv2.INTER_LINEAR)
        img_np = np.array(img)
        img_np = self.undistortRectify(img_np, map_calib_file, left_True)
        img_np = np.array(img_np).astype(np.uint8)
        img = torch.from_numpy(img_np).permute(2, 0, 1).float()
        return img[None].to(self.device)

    def reconstruct_2d_to_3d(self, ImL, mask_img, disparity, map_calib_file):
        cv_file = cv2.FileStorage()
        cv_file.open(map_calib_file, cv2.FileStorage_READ)
        ImL = self.undistortRectify(ImL, map_calib_file, True)
        Q1_map = cv_file.getNode('q').mat()
        cv_file.release()
        h, w = disparity.shape
        # Q = np.float32([
        #     [1, 0, 0, -w / 2.0],
        #     [0, -1, 0, h / 2.0],
        #     [0, 0, 0, Q1_map[2, 3]],
        #     [0, 0, -Q1_map[3, 2], Q1_map[3, 3]]
        # ])
        valid_disp_mask = disparity > 0
        if mask_img.dtype != bool:
            mask_img = mask_img > 0
        combined_mask = np.logical_and(valid_disp_mask, mask_img)
        points_3D = cv2.reprojectImageTo3D(disparity, Q1_map)
        colors = cv2.cvtColor(ImL, cv2.COLOR_BGR2RGB)
        colors = colors / 255
        out_points = points_3D[combined_mask].reshape(-1, 3)
        colors = colors/255
        out_colors = colors[combined_mask].reshape(-1, 3)
        return out_points , out_colors
    def calculate_2d_depth(self, image_L, post_mask, disp_pr, baseline = 50, fx= 382.3565, fy =  382.9777, cx1 = 323.724, cy = 238.944, cx2 = 323.3898):
        # Calibration
        # fx, fy, cx1, cy = 615.75, 616.02, 329.27, 244.46
        # fx, fy, cx1, cy = 382.3565, 382.9777, 323.724, 238.944
        # cx2 = 323.3898
        # baseline= 50 # in millimeters #50mm
        # inverse-project
        depth = ((fx * baseline) / (-disp_pr + (cx2 - cx1))) 
        H, W = depth.shape
        xx, yy = np.meshgrid(np.arange(W), np.arange(H))
        points_grid = np.stack(((xx-cx1)/fx, (yy-cy)/fy, np.ones_like(xx)), axis=0) * depth
        mask = np.ones((H, W), dtype=bool)
        # Remove flying points
        mask[1:][np.abs(depth[1:] - depth[:-1]) > 1] = False
        mask[:,1:][np.abs(depth[:,1:] - depth[:,:-1]) > 1] = False
        mask = np.logical_and(mask, post_mask.astype(np.uint8))
        points = points_grid.transpose(1,2,0)[mask]
        colors = image_L[mask].astype(np.float64) / 255
        return points, colors
    def crop_image_by_bbox(self,image, bbox):
        if not isinstance(bbox, (list, tuple, np.ndarray)) or len(bbox) != 4:
            print("Error: Invalid bounding box format. Expected [x_min, y_min, x_max, y_max].")
            return None

        x_min, y_min, x_max, y_max = map(int, bbox)

        height, width = image.shape[:2]

        if x_min < 0 or y_min < 0 or x_max > width or y_max > height:
            print("Warning: Bounding box is out of image bounds.")
            return None

        cropped_image = image[y_min:y_max, x_min:x_max]
        return cropped_image
    def convert_box(self, original_box, original_size=(640, 640), new_size=(640, 480)):
        """
        Converts bounding box coordinates from one image size to another.

        Args:
            original_box (tuple): Bounding box in (x_min, y_min, x_max, y_max) format for the original image size.
            original_size (tuple): Original image size (width, height).
            new_size (tuple): New image size (width, height).

        Returns:
            tuple: Converted bounding box (x_min, y_min, x_max, y_max) for the new image size.
        """
        original_width, original_height = original_size
        new_width, new_height = new_size
        
        # Scaling factors
        scale_x = new_width / original_width
        scale_y = new_height / original_height
        
        # Convert the bounding box coordinates
        x_min, y_min, x_max, y_max = original_box
        new_x_min = x_min * scale_x
        new_y_min = y_min * scale_y
        new_x_max = x_max * scale_x
        new_y_max = y_max * scale_y
        
        return np.array([new_x_min, new_y_min, new_x_max, new_y_max])
    def depth_estimation(self, im_L, im_R, im_color, map_calib_file):
        if im_color is None :
            print("Warning: No frames captured")
            return None
        
        with torch.no_grad():
            # ---- DEFOM Stereo model ---------------
            image1 = self.load_image(im_L, map_calib_file, True)
            image2 = self.load_image(im_R, map_calib_file, False)
            padder = InputPadder(image1.shape, divis_by=32)
            image1, image2 = padder.pad(image1, image2)
            with torch.no_grad():
                disp_pr = self.defom_stereo_model(image1, image2, test_mode=True)
            disp_pr = padder.unpad(disp_pr).cpu().squeeze().numpy()
            # -------------- YOLOE -------------------
            image_color = cv2.cvtColor(im_L, cv2.COLOR_BGR2RGB)
            un_image = self.undistortRectify(image_color, map_calib_file, True)
            re_image = cv2.resize(un_image, (640,640), cv2.INTER_NEAREST)
            r = self.see_any_thing.predict(re_image, save=False)[0].to(self.device)
            pred_inf =[]
            if r.masks is not None:
                seg_ob = Results(orig_img=r.orig_img, path=r.path, names=r.names, boxes=r.boxes.data, masks=r.masks.data).plot()
                cv2.imshow("Result", seg_ob)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                results = self.apply_nms(r)
                pred_boxes = results.boxes.xyxy.cpu().numpy()
                clasess = results.boxes.data.cpu().numpy()
                pred_masks = results.masks.data.cpu().numpy()
                for _mask, box, cls_name in zip(pred_masks, pred_boxes[:, :4], clasess[:, 5]):
                    # _mask = (_mask * 255).astype(np.uint8)
                    _mask = cv2.resize(_mask, (640, 480), interpolation=cv2.INTER_NEAREST)
                    post_mask = self._mask_processing(_mask)
                    points, colors = self.reconstruct_2d_to_3d(image_color, post_mask, disp_pr, map_calib_file)
                    # points, colors = self.calculate_2d_depth(un_image, post_mask, disp_pr)
                    pcd = self.compute_normal(points, colors )
                    re_box = self.convert_box(box)
                    image_crop = self.crop_image_by_bbox(un_image, re_box)
                    pred_inf.append([pcd, image_crop, cls_name])
            else:
                pred_inf = None
            return pred_inf