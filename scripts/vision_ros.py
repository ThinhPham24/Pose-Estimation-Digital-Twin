from utils.d435_capture import initialize, capture_frame, stereo_realsense
from multimodal import ObjectPointCloudGenerator
from utils.similarity_2D import calculate_vgg19_pytorch_similarity
from utils.matching_3D import PointCloudRegistrator
from generate.capture import CADViewGenerator
import cv2
import os 
import numpy as np
import open3d as o3d
from utils.draw_bbox import draw_detections
from PIL import Image
import trimesh
def visualize(rgb, pred_rot, pred_trans, model_points, K, save_path):
    img = draw_detections(rgb, pred_rot, pred_trans, model_points, K, color=(255, 0, 0))
    img = Image.fromarray(np.uint8(img))
    img.save(save_path)
    prediction = Image.open(save_path)
    # concat side by side in PIL
    rgb = Image.fromarray(np.uint8(rgb))
    img = np.array(img)
    concat = Image.new('RGB', (img.shape[1] + prediction.size[0], img.shape[0]))
    concat.paste(rgb, (0, 0))
    concat.paste(prediction, (img.shape[1], 0))
    return concat
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

try:
    '''
    1. Capture image from Realsene
    2. Init all models
    3. Inference the seeing anything model
    4. Inference the depth anything model
    5. Check similarity score
    6. Match the Point cloud registor
    7. Refine the real coordinate of realsense camera
    8. Postprocessing pose estimation
    '''
    map_class = {'0': "obj1", '1': 'obj2'}
    map_views = {'back_top': 'back_view', 'front_top':'front_view', 'left_top':'left_view', 'right_top': 'right_view', 'Top': 'top_view'}
    defom_ckpt = "/home/airlab/Desktop/DigitalTwin_PoseEstimation/scripts/checkpoints/defomstereo_vitl_sceneflow.pth"
    yoloe_ckpt = "ultralytics/pretrain/yoloe-v8l-seg.pt"
    map_calibrate = '/home/airlab/Desktop/DigitalTwin_PoseEstimation/scripts/Calibrated/stereoMap.txt'
    output_dir = '/home/airlab/Desktop/DigitalTwin_PoseEstimation/'
    generator_pc = ObjectPointCloudGenerator(defom_ckpt, yoloe_ckpt)
   
    while True:
        # Inital Realsense camera
        real_camera = stereo_realsense()
        im_color, imL, imR, _, intrinsics = real_camera.capture_frame()
        print("Size img color", im_color.shape)
        print("Size img color L", imL.shape)
        print("Size img color R", imR.shape)

        # cv2.imshow("image", im_color)
        # cv2.imshow("imL", imL)
        # cv2.imshow("imR", imR)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # if cv2.waitKey(1) & 0xFF == 27:  # ESC
        #     break
        object_infs = generator_pc.depth_estimation(imL, imR, im_color, map_calibrate)
        if object_infs is None:
            print("No object detections")
            pass 
        else:
            rotations  = []
            translations = []
            for pcd, img, cls  in object_infs:
                # Map the class name
                cls_str = str(int(cls))
                object_name = map_class[cls_str]
                # o3d.visualization.draw_geometries([pcd])
                # cv2.imshow("Images", img)
                # cv2.waitKey(0)
                # Generate the partial CAD model for each view  - five views - Can create before           
                cad_model_path = f'/home/airlab/Desktop/DigitalTwin_PoseEstimation/data/ply_models/{object_name}.ply'
                output_dir_cad = f"/home/airlab/Desktop/DigitalTwin_PoseEstimation/data/partial_point_clouds/{object_name}/"
                generator_cad = CADViewGenerator(cad_model_path, output_dir_cad)
                generator_cad.generate_partial_point_clouds()
                generator_cad.capture_cad_views_with_added_color()
                # Calculate cosine similarity - Using VVG19 extracted feature
                similarity_results = calculate_vgg19_pytorch_similarity(img, output_dir_cad)
                # # Print similarity scores
                # for filename, score in similarity_results.items():
                #     print(f"Similarity with {filename}: {score}")
                # # Find the most similar image
                if similarity_results:
                    most_similar_image = max(similarity_results, key=similarity_results.get)
                    print(f"\nMost similar image: {most_similar_image} (Similarity: {similarity_results[most_similar_image]})")
                    threshold = 0.12
                    if float(similarity_results[most_similar_image]) < threshold:
                        print("Not into dataset")
                        continue
                    else:
                        #PWS_ICP_KTree matching
                        filename, extension = os.path.splitext(most_similar_image)
                        view_name = map_views[filename]
                        # Need to modify for cutting partial point cloud of each view from camera, respactively.
                        # target_path = f'/home/airlab/Desktop/DigitalTwin_PoseEstimation/data/partial_point_clouds/{object_name}/{view_name}.ply'
                        target_path = f'/home/airlab/Desktop/DigitalTwin_PoseEstimation/data/partial_point_clouds/{object_name}/top_view.ply'
                        init_target_path = f'/home/airlab/Desktop/DigitalTwin_PoseEstimation/data/ply_models/{object_name}.ply'
                        registrator = PointCloudRegistrator(pcd, target_path, init_target_path)
                        transformation = registrator.register_point_clouds()
                        inv_transformation = np.linalg.inv(transformation)
                        rotations.append(inv_transformation[:3, :3])
                        translations.append(inv_transformation[:3, 3])
                        if transformation is not None:
                            print("Transformation matrix:", transformation)
                            print("Translate", transformation[:3,3])
                            print("Rotation", transformation[:3,:3])
                            registrator.visualize_transform(pcd, transformation)
            K = np.array([[intrinsics.fx, 0, intrinsics.ppx],
                    [0, intrinsics.fy, intrinsics.ppy],
                    [0, 0, 1]]).reshape(3,3)
            # K = np.array([[615.75, 0, 329.27],
            #         [0, 616.02, 244.46],
            #         [0, 0, 1]]).reshape(3,3)
            # 382.3565, 382.9777, 323.724, 238.944
            # fx, fy, cx1, cy = 615.75, 616.02, 329.27, 244.46
            print("K", K)
            print("=> visualizating ...")
            save_path = os.path.join(f"{output_dir}/Pose_results", 'result.png')
            mesh = trimesh.load_mesh(init_target_path)
            model_points = mesh.sample(1024).astype(np.float32)
            vis_img = visualize(imL, rotations, translations, model_points, K, save_path)
            # Refine the transformation matrix to real scenario
            # Convert to realsencamera
            # Convert camera coordinate to robot coordinate
            # Publish to ROS       
        real_camera.release()

finally:
    # Stop the RealSense pipeline when done
    real_camera.release()
    cv2.destroyAllWindows()
 
    
   
                 
       



