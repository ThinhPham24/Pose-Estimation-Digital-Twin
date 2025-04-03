from utils.d435_capture import initialize, capture_frame
from multimodel import ObjectPointCloudGenerator
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
if __name__ == "__main__":
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
    depth_anything_path = '/home/airlab/Desktop/DigitalTwin_PoseEstimation/scripts/depth_anything_v2/checkpoints/depth_anything_v2_vitl.pth'
    yoloe_path = "ultralytics/pretrain/yoloe-v8l-seg.pt"
    generator_pc = ObjectPointCloudGenerator(depth_anything_path, yoloe_path)

    # Inital Realsense camera
    '''

    '''

    image = cv2.imread("/home/airlab/Desktop/DigitalTwin_PoseEstimation/data/images/Test3.png")
    output_dir = '/home/airlab/Desktop/DigitalTwin_PoseEstimation/'
    re_image = cv2.resize(image, (640,480), cv2.INTER_AREA)
    object_infs = generator_pc.depth_estimation(image)
    if object_infs is None:
        print("No object detections")
        pass 
    else:
        for pcd, img, cls  in object_infs:
            # Map the class name
            cls_str = str(int(cls))
            object_name = map_class[cls_str]
            # Generate the partial CAD model for each view  - five views            
            cad_model_path = f'/home/airlab/Desktop/DigitalTwin_PoseEstimation/data/ply_models/{object_name}.ply'
            output_dir_cad = f"/home/airlab/Desktop/DigitalTwin_PoseEstimation/data/partial_point_clouds/{object_name}/"
            generator_cad = CADViewGenerator(cad_model_path, output_dir_cad)
            generator_cad.generate_partial_point_clouds()
            generator_cad.capture_cad_views_with_added_color()
            # Calculate cosine similarity - Using VVG19 extracted feature
            similarity_results = calculate_vgg19_pytorch_similarity(img, output_dir_cad)
            # cv2.imshow("Images", img)
            # cv2.waitKey(0)s
            # # Print similarity scores
            # for filename, score in similarity_results.items():
            # print(f"Similarity with {filename}: {score}")
            # Find the most similar image
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
                    # source_path = "/home/airlab/Desktop/DigitalTwin_PoseEstimation/data/source/d435_on_ur10e_arm/source_ob1_0.ply"
                    # Need to modify for cutting partial point cloud of each view from camera, respactively.
                    # target_path = f'/home/airlab/Desktop/DigitalTwin_PoseEstimation/data/partial_point_clouds/{object_name}/{view_name}.ply'
                    target_path = f'/home/airlab/Desktop/DigitalTwin_PoseEstimation/data/partial_point_clouds/{object_name}/top_view.ply'
                    init_target_path = f'/home/airlab/Desktop/DigitalTwin_PoseEstimation/data/ply_models/{object_name}.ply'
                    registrator = PointCloudRegistrator(pcd, target_path, init_target_path)
                    transformation = registrator.register_point_clouds()
                    if transformation is not None:
                        print("Transformation matrix:", transformation)
                        print("Translate", transformation[:3,3])
                        print("Rotation", transformation[:3,:3])
                        registrator.visualize_registration(transformation)
                 
                    K = np.array([[470.4,0,320], [0,470.4,320], [0,0,1]]).reshape(3, 3)
                    print("K", K)
                    print("=> visualizating ...")
                    save_path = os.path.join(f"{output_dir}/Pose_results", 'vis_pem.png')
                    mesh = trimesh.load_mesh(init_target_path)
                    model_points = mesh.sample(1024).astype(np.float32)
                    vis_img = visualize(re_image, transformation[:3,:3], transformation[:3,3], model_points/2, K, save_path)
                    vis_img.save(save_path)
                    # Refine the transformation matrix to real scenario
                    # Convert to realsencamera
                    # Convert camera coordinate to robot coordinate
                    # Publish to ROS           
            else:
                print("No images found or errors occurred.")
                continue


