from ultralytics import YOLOE
import numpy as np
from ultralytics.models.yolo.yoloe.predict_vp import YOLOEVPSegPredictor

model = YOLOE("pretrain/yoloe-v8l-seg.pt")

# Handcrafted shape can also be passed, please refer to app.py
# Multiple boxes or handcrafted shapes can also be passed as visual prompt in an image
visuals = dict(
    bboxes=np.array(
        [
            [298.2, 190.1, 375.2, 240.1],  # For person
            # [400, 238, 480, 307],
            # [320,304, 392, 365],  # For glasses
        ],

    ),
    cls=np.array(
        [
            0,  # cube
        ]
    )
)

source_image = 'ultralytics/assets/color2.png'
image_size = 640
model.predict(source_image, imgsz=image_size, save=True, prompts=visuals, predictor=YOLOEVPSegPredictor)

# Prompts in different images can be passed
# Please set a smaller conf for cross-image prompts
# model.predictor = None  # remove VPPredictor
target_image = 'ultralytics/assets/color3.png'
model.predict(source_image, imgsz=image_size, prompts=visuals, predictor=YOLOEVPSegPredictor,
              return_vpe=True)
model.set_classes(["cube"], model.predictor.vpe)
model.predictor = None  # remove VPPredictor
model.predict(target_image, save=True)
