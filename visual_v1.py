from detectron2.data import (
    MetadataCatalog,
    build_detection_test_loader
)
from detectron2.utils.visualizer import Visualizer
from detectron2.engine import DefaultPredictor
import cv2

from tqdm import tqdm
import time

def visual(cfg):
    visual_what = 'davis_val'
    data_loader = build_detection_test_loader(cfg, visual_what)
    predictor = DefaultPredictor(cfg)

    for idx, inputs in tqdm(enumerate(data_loader)):
        image = cv2.imread(inputs[0]['file_name'])
        outputs = predictor(image)

        v = Visualizer(image[:,:,::-1],metadata=MetadataCatalog.get(visual_what),scale=0.8)
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))

        cv2.imwrite('visual_result/{}.jpg'.format(time.time()),v.get_image()[:,:,::-1])

