import cv2
import numpy as np
from PIL import Image
from utils.draw_bbox import draw_detections
def visualize(rgb, pred_rots, pred_trans, model_points, K, save_path):
    img = draw_detections(rgb, pred_rots, pred_trans, model_points, K, color=(255, 255, 255))
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