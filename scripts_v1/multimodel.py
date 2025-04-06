import pyrealsense2 as rs
import numpy as np
import cv2
import torch
import open3d as o3d
from depth_anything_v2.dpt import DepthAnythingV2
from PIL import Image
from ultralytics import YOLOE
from ultralytics.models.yolo.yoloe.predict_vp import YOLOEVPSegPredictor
from ultralytics.engine.results import Results
from ultralytics.engine.results import Boxes
class ObjectPointCloudGenerator:
    def __init__(self, depth_anything_path, yoloe_path, device="cuda:0"):
        self.device = torch.device(device) if torch.cuda.is_available() else torch.device("cpu")
        self.depth_any_thing = self._load_depth_anything(depth_anything_path)
        self.see_any_thing = self._load_yoloe(yoloe_path)
    def _load_depth_anything(self, model_path):
        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }
        encoder = 'vitl'
        depth_any_thing = DepthAnythingV2(**model_configs[encoder])
        depth_any_thing.load_state_dict(torch.load(model_path, map_location=self.device))
        return depth_any_thing.to(self.device).eval()

    def _load_yoloe(self, model_path, source_image = 'ultralytics/assets/Test3.png', class_name = ["cude"]):
        see_any_thing = YOLOE(model_path).to(self.device)
        visuals = dict(
            # bboxes=np.array([[298.2, 190.1, 375.2, 240.1]]),
            bboxes=np.array([[178.2, 218.1, 365.2, 372.1]]),
            cls=np.array([0])
        )
        image_size = 640
        see_any_thing.predict(source_image, imgsz=image_size, prompts=visuals, predictor=YOLOEVPSegPredictor, return_vpe=True)
        see_any_thing.set_classes(class_name, see_any_thing.predictor.vpe)
        see_any_thing.predictor = None
        return see_any_thing
    def apply_nms(self, results, iou_threshold=0.5):
        """
        Applies Non-Maximum Suppression (NMS) to filter detection results.

        Args:
            results (ultralytics.engine.results.Results): Ultralytics Results object.
            iou_threshold (float): Intersection over Union (IoU) threshold for NMS.

        Returns:
            ultralytics.engine.results.Results: Filtered Results object.
        """
        boxes = results.boxes.xyxy.cpu().numpy()
        scores = results.boxes.conf.cpu().numpy()
        masks = results.masks.data.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy()

        if len(boxes) == 0:
            return results  # No detections, return original results

        # Apply NMS
        selected_indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), score_threshold=0.1, nms_threshold=iou_threshold)

        if len(selected_indices) > 0:
            selected_indices = selected_indices.flatten().tolist()
            filtered_boxes = boxes[selected_indices]
            filtered_scores = scores[selected_indices]
            filtered_masks = masks[selected_indices]
            filtered_classes = classes[selected_indices]

            # Reconstruct a filtered results object
            filtered_boxes_tensor = torch.tensor(filtered_boxes, device=results.boxes.xyxy.device)
            filtered_scores_tensor = torch.tensor(filtered_scores, device=results.boxes.conf.device)
            filtered_classes_tensor = torch.tensor(filtered_classes, device=results.boxes.cls.device)

            # Create new Boxes object
            filtered_boxes_data = torch.cat([filtered_boxes_tensor, filtered_scores_tensor.unsqueeze(1), filtered_classes_tensor.unsqueeze(1)], dim=1)
            filtered_boxes_obj = Boxes(filtered_boxes_data, orig_shape=results.boxes.orig_shape)

            results.boxes = filtered_boxes_obj
            results.masks.data = torch.tensor(filtered_masks, device=results.masks.data.device)

        else:
            # No detections after NMS, empty results.
            empty_boxes_data = torch.tensor([], device=results.boxes.xyxy.device).reshape(0, 6)
            empty_boxes = Boxes(empty_boxes_data, orig_shape=results.boxes.orig_shape)
            results.boxes = empty_boxes
            results.masks.data = torch.tensor([], device=results.masks.data.device).reshape(0, results.masks.data.shape[1], results.masks.data.shape[2], results.masks.data.shape[3])

        return results

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
    def generate_point_cloud(self, final_depth, image_color, z_thresmax = 1):
        """
        Generate 3D point cloud from depth and RGB image using camera intrinsics,
        and filter out points with depth greater than z_threshold.

        Args:
            final_depth (np.ndarray): Depth map (H x W), values in millimeters.
            image_color (np.ndarray): RGB image (H x W x 3), same resolution as depth.
            z_threshold (float): Maximum z (depth in meters) to keep.

        Returns:
            points (np.ndarray): Nx3 array of filtered 3D points.
            colors (np.ndarray): Nx3 array of filtered RGB colors.
        """

        # Intrinsics (example for RealSense)
        # fx, fy = 615.75, 616.02
        # cx, cy = 329.27, 244.46
        fx, fy = 470.4, 470.4
        cx, cy = 320, 320

        x, y = np.meshgrid(np.arange(640), np.arange(640))

        x = (x - cx) / fx
        y = (y - cy) / fy
        z = final_depth.astype(np.float32) / 1000.0  # mm → meters

        X = x * z
        Y = y * z
        Z = z
        z_thresmin = 0.3
        points = np.stack((X, Y, Z), axis=-1).reshape(-1, 3)
        colors = image_color.astype(np.float32).reshape(-1, 3) / 255.0

        # Flatten Z for masking
        Z_flat = Z.flatten()

        # Apply Z threshold mask
        valid_mask = (Z_flat > z_thresmin) & (Z_flat < z_thresmax)
        points = points[valid_mask]
        colors = colors[valid_mask]
        return [points, colors]
    def compute_normal(self, points, colors):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)
        mask = points[:, 2] > points[:, 2].min()
        pcd.points = o3d.utility.Vector3dVector(points[mask])
        pcd.colors = o3d.utility.Vector3dVector(colors[mask])
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=10))
        pcd.orient_normals_consistent_tangent_plane(k=5)
        # o3d.visualization.draw_geometries([pcd],window_name = "compute_normal" )
        # o3d.io.write_point_cloud(output_ply_path.format(index), pcd, write_ascii=True)
        return pcd
    def convert_box(self, original_box, original_size=(640, 640), new_size=(640, 640)):
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
    def _mask_processing(self, mask):
        h,w = mask.shape[:2]
        binary_image = np.zeros((h, w), dtype=np.uint8)
        m = (mask*255).astype(np.uint8)
        m = np.squeeze(m)
        contours,_ = cv2.findContours(m,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        mCnt = max(contours, key = cv2.contourArea)
        _mask = cv2.drawContours(binary_image, [mCnt], -1, 255, cv2.FILLED)
        kernel = np.ones((3, 3), np.uint8) 
        post_mask = cv2.erode(_mask, kernel, iterations=1) 
        post_mask = cv2.dilate(_mask, kernel, iterations=1) 
        return post_mask
    def depth_estimation(self, image_color):
        if image_color is None :
            print("Warning: No frames captured")
            return None
        image_color = cv2.cvtColor(image_color, cv2.COLOR_BGR2RGB)
        SAT_image = cv2.resize(image_color, (640, 640),cv2.INTER_NEAREST)
        pred_depth = self.depth_any_thing.infer_image(SAT_image)
        # resized_pred_depth = np.array(Image.fromarray(pred_depth).resize((640, 640), Image.NEAREST))
        # disparity = resized_pred_depth.astype(np.float32) / 16
        pred_depth[pred_depth == 0] = 1e-6
        r = self.see_any_thing.predict(SAT_image, save=False)[0].to(self.device)
        # seg_ob = Results(orig_img=r.orig_img,path=r.path,names= r.names,boxes=r.boxes.data, masks=r.masks.data).plot()
        # cv2.imshow("Result", seg_ob)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        pred_inf =[]
        if r.masks is not None:
            results = self.apply_nms(r)
            pred_boxes = results.boxes.xyxy.cpu().numpy()
            clasess = results.boxes.data.cpu().numpy()
            pred_masks = results.masks.data.cpu().numpy()
            for mask, box, cls_name in zip(pred_masks, pred_boxes[:, :4], clasess[:, 5]):
                # _mask = cv2.resize(mask, (640, 480), interpolation=cv2.INTER_NEAREST)
                mask = self._mask_processing(mask)
                depth_map = cv2.bitwise_and(pred_depth, pred_depth, mask=mask.astype(np.uint8))
                point_clouds = self.generate_point_cloud(depth_map, SAT_image, z_thresmax = 1)
                pcd = self.compute_normal(point_clouds[0], point_clouds[1])
                re_box = self.convert_box(box)
                image_crop = self.crop_image_by_bbox(SAT_image, re_box)
                pred_inf.append([pcd, image_crop, cls_name])
        else:
            pred_inf = None
        return pred_inf

# if __name__ == "__main__":
#     depth_anything_path = R'/home/airlab/Desktop/DigitalTwin_PoseEstimation/checkpoints/depth_anything_v2_vitl.pth'
#     yoloe_path = "pretrain/yoloe-v8l-seg.pt"
#     output_ply_path = R"/home/airlab/Desktop/DigitalTwin_PoseEstimation/data/source/d435_on_ur10e_arm/source_ob1_{}.ply"
#     generator = ObjectPointCloudGenerator(depth_anything_path, yoloe_path)
#     image = cv2.imread("/home/airlab/Desktop/DigitalTwin_PoseEstimation/scripts/Test.png")
#     object_name = generator.depth_estimation(image)
#     for pcd, img, cls  in object_name:
#         o3d.visualization.draw_geometries([pcd]) 
#         cv2.imshow("Image", img)
#         cv2.waitKey(0)