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

    def _load_yoloe(self, model_path, source_image = 'ultralytics/assets/color2.png', class_name = ["cude"]):
        see_any_thing = YOLOE(model_path).to(self.device)
        visuals = dict(
            bboxes=np.array([[298.2, 190.1, 375.2, 240.1]]),
            # bboxes=np.array([[96.2, 237.1, 224.2, 321.1]]),
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
    def generate_point_cloud(self, final_depth, image_color):
        cx, cy = 320, 320
        fx, fy = 470.4, 470.4
        # fx, fy = 617.17, 615.04
        x, y = np.meshgrid(np.arange(640), np.arange(640))
        x = ((x - cx) / fx) * 1.4
        y = ((y - cy) / fy) * 1.4
        z = final_depth / 100
        points = np.stack((np.multiply(x, z), np.multiply(y, z), z), axis=-1).reshape(-1, 3)
        colors = np.array(image_color).reshape(-1, 3) / 255.0
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
        # o3d.io.write_point_cloud(output_ply_path.format(index), pcd, write_ascii=True)
        return pcd
    def depth_estimation(self, image_color):
        if image_color is None :
            print("Warning: No frames captured")
            return None
        image_color = cv2.cvtColor(image_color, cv2.COLOR_BGR2RGB)
        resize_image_color = cv2.resize(image_color, (640, 640))
        pred_depth = self.depth_any_thing.infer_image(resize_image_color)
        resized_pred_depth = np.array(Image.fromarray(pred_depth).resize((640, 640), Image.NEAREST))
        disparity = resized_pred_depth.astype(np.float32) / 16
        disparity[disparity == 0] = 1e-6
        r = self.see_any_thing.predict(resize_image_color, save=False)[0].to(self.device)
        seg_ob = Results(orig_img=r.orig_img,path=r.path,names= r.names,boxes=r.boxes.data, masks=r.masks.data).plot()
        cv2.imshow("Result", seg_ob)
        cv2.waitKey(0)
        results = self.apply_nms(r)
        pred_inf =[]
        if results.masks is not None:
            pred_boxes = results.boxes.xyxy.cpu().numpy()
            clasess = results.boxes.data.cpu().numpy()
            pred_masks = results.masks.data.cpu().numpy()
            for mask, box, cls_name in zip(pred_masks, pred_boxes[:, :4], clasess[:, 5]):
                final_depth = cv2.bitwise_and(disparity, disparity, mask=mask.astype(np.uint8))
                point_clouds = self.generate_point_cloud(final_depth, resize_image_color)
                pcd = self.compute_normal(point_clouds[0], point_clouds[1])
                image_crop = self.crop_image_by_bbox(resize_image_color, box)
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