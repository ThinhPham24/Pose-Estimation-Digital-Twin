import pyrealsense2 as rs
import numpy as np
import cv2
import torch
import open3d as o3d
from ultralytics import YOLOE
from ultralytics.models.yolo.yoloe.predict_vp import YOLOEVPSegPredictor
from ultralytics.engine.results import Results
from ultralytics.engine.results import Boxes
from utils.matching_3D import PointCloudRegistrator
from PIL import Image
import trimesh
import os
from utils.draw_bbox import draw_detections


def _load_yoloe(model_path, source_image='ultralytics/assets/Test3.png', class_name=["cude"]):
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    see_any_thing = YOLOE(model_path).to(device)
    visuals = dict(
        bboxes=np.array([[178.2, 218.1, 365.2, 372.1]]),
        cls=np.array([0])
    )
    image_size = 640
    see_any_thing.predict(source_image, imgsz=image_size, prompts=visuals, predictor=YOLOEVPSegPredictor, return_vpe=True)
    see_any_thing.set_classes(class_name, see_any_thing.predictor.vpe)
    see_any_thing.predictor = None
    return see_any_thing

def apply_nms(results, iou_threshold=0.5):
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

def compute_normal(points, colors):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    # mask = points[:, 2] > points[:, 2].min()
    # pcd.points = o3d.utility.Vector3dVector(points[mask])
    # pcd.colors = o3d.utility.Vector3dVector(colors[mask])
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=10))
    pcd.orient_normals_consistent_tangent_plane(k=5)
    return pcd

def draw_contours_from_mask(image, mask, contour_color=(0, 255, 0), contour_thickness=1):
    if len(mask.shape) > 2:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if contours:
        max_cont = max(contours, key=cv2.contourArea)
        cv2.drawContours(image, [max_cont], -1, contour_color, contour_thickness)

    return image

def draw_scaled_bbox(image, bbox_640x640, bbox_color=(255, 0, 0), bbox_thickness=2):
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

def get_colored_point_cloud_from_bbox(depth_image, color_image, intrinsics, bbox):
    x_min, y_min, x_max, y_max = bbox
    points_3d = []
    colors_rgb = []

    for y in range(int(y_min), int(y_max)):
        for x in range(int(x_min), int(x_max)):
            depth = depth_image[y, x]
            if depth > 0:
                point_3d = rs.rs2_deproject_pixel_to_point(intrinsics, [x, y], depth)
                points_3d.append(np.array(point_3d)/1000)
                colors_rgb.append(color_image[y, x] / 255.0)

    return np.array(points_3d), np.array(colors_rgb)

def visualize_point_cloud(points, colors):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd])

def visualize(rgb, pred_rot, pred_trans, model_points, K, save_path):
    img = draw_detections(rgb, pred_rot, pred_trans, model_points, K, color=(255, 0, 0))
    img = Image.fromarray(np.uint8(img))
    img.save(save_path)
    prediction = Image.open(save_path)
    rgb = Image.fromarray(np.uint8(rgb))
    img = np.array(img)
    concat = Image.new('RGB', (img.shape[1] + prediction.size[0], img.shape[0]))
    concat.paste(rgb, (0, 0))
    concat.paste(prediction, (img.shape[1], 0))
    return concat

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
output_dir = '/home/airlab/Desktop/DigitalTwin_PoseEstimation/'

def capture_point_cloud_no_scale():
    try:
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        pipeline.start(config)
        frames = pipeline.wait_for_frames()
        align = rs.align(rs.stream.color)
        aligned_frames = align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        if not depth_frame or not color_frame:
            print("Could not retrieve depth or color frame.")
            return None
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        depth_intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics

        cv2.imshow("Image", color_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        see_any_thing = _load_yoloe("ultralytics/pretrain/yoloe-v8l-seg.pt")
        r = see_any_thing.predict(color_image, save=False)[0].to(device)
        seg_ob = Results(orig_img=r.orig_img, path=r.path, names=r.names, boxes=r.boxes.data, masks=r.masks.data).plot()
        cv2.imshow("Result", seg_ob)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        results = apply_nms(r)
        # pc = rs.pointcloud()
        # points = pc.calculate(depth_frame)
        # vtx = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, 3)
        # valid_mask = vtx[:, 2] < 0.5
        # valid_vtx = vtx[valid_mask]
        # valid_color_vtx = color_image.reshape(-1, 3)[valid_mask]
        # point_cloud = o3d.geometry.PointCloud()
        # point_cloud.points = o3d.utility.Vector3dVector(valid_vtx)
        # point_cloud.colors = o3d.utility.Vector3dVector(valid_color_vtx / 255.0)
        # o3d.visualization.draw_geometries([point_cloud])

        if results.masks is not None and results.masks.data.numel() > 0: # Check if masks exist
            pred_boxes = results.boxes.xyxy.cpu().numpy()
            clasess = results.boxes.data.cpu().numpy()
            pred_masks = results.masks.data.cpu().numpy()
            for _mask, box, cls_name in zip(pred_masks, pred_boxes[:, :4], clasess[:, 5]):
                # _mask = (_mask * 255).astype(np.uint8)
                # mask = cv2.resize(_mask, (640, 480), interpolation=cv2.INTER_LINEAR)
                points_3d, colors_rgb = get_colored_point_cloud_from_bbox(depth_image, color_image, depth_intrinsics, box)
                # visualize_point_cloud(points_3d, colors_rgb)
                points = np.array(points_3d)
                colors = np.array(colors_rgb)
                pcd = compute_normal(points, colors)
                # target_path = '/home/airlab/Desktop/DigitalTwin_PoseEstimation/data/partial_point_clouds/obj1/top_view.ply'
                target_path = '/home/airlab/Desktop/DigitalTwin_PoseEstimation/data/ply_models/obj1.ply'
                init_target_path = '/home/airlab/Desktop/DigitalTwin_PoseEstimation/data/ply_models/obj1.ply'
                registrator = PointCloudRegistrator(pcd, target_path, init_target_path)
                transformation = registrator.register_point_clouds()
                if transformation is not None:
                    print("Transformation matrix:", transformation)
                    print("Translate", transformation[:3, 3])
                    print("Rotation", transformation[:3, :3])
                    registrator.visualize_registration(transformation)

                K = np.array([[depth_intrinsics.fx, 0, depth_intrinsics.ppx],
                      [0, depth_intrinsics.fy, depth_intrinsics.ppy],
                      [0, 0, 1]])
                print("K", K)
                print("=> visualizating ...")
                save_path = os.path.join(f"{output_dir}/Pose_results", 'vis_pem.png')
                mesh = trimesh.load_mesh(init_target_path)
                model_points = mesh.sample(2048).astype(np.float32)
                vis_img = visualize(color_image, transformation[:3, :3], transformation[:3, 3], model_points, K, save_path)
                vis_img.save(save_path)
                del model_points
                del transformation

        pipeline.stop()

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

if __name__ == "__main__":
    point_cloud = capture_point_cloud_no_scale()
