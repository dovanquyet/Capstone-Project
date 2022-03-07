import cv2
import os
import shutil
import sys
import time
import argparse
import numpy as np

from config import config

from core.visualize.utils import draw_inferences
from core.logging import configure_logger
from core.models import models
from core.utils.bbox import crop_bbox

from core.objects import DetectedObject

logger = configure_logger()

def handle_image(detector, helmet_classifier, jacket_classifier,
                 frame, output_dir, file_name, save_crop, 
                 save_frame=True, frame_num=1):
    t1 = time.time()
    detections, labels = detector.infer(frame)
        
    inferences = []
    for i, (detection, label) in enumerate(zip(detections, labels)):
        x1, y1, x2, y2, angle, score = detection
        bbox = list(map(int, [x1, y1, x2, y2]))
        bbox.append(np.float64(angle))
        detected_object = DetectedObject(bbox, confidence=np.float64(score))
        region = crop_bbox(frame, detected_object)
        helmet_label = helmet_classifier.infer(region)
        jacket_label = jacket_classifier.infer(region)
        inferences.append([detected_object, helmet_label, jacket_label])
        
        if save_crop:
            os.makedirs('crops', exist_ok=True)
            os.makedirs(os.path.join("crops", f"{file_name[:-4]}"), exist_ok=True)
            cv2.imwrite(os.path.join("crops", f"{file_name[:-4]}", f"{frame_num}_{i}.jpg"), region)
            
    frame = draw_inferences(frame.copy(), inferences)
    if save_frame:
        cv2.imwrite(output_dir, frame)
        logger.info(f"Frame 1/1 - Time: {time.time()- t1}s")
    else:
        return frame

def handle_video(detector, helmet_classifier, jacket_classifier,
                 input_dir, output_dir, file_name, save_crop):
    cap = cv2.VideoCapture(input_dir)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    
    fourcc = cv2.VideoWriter_fourcc(*config.video_encoder)
    writer = cv2.VideoWriter(output_dir, fourcc, fps, (frame_w, frame_h))
    
    for frame_num in range(total_frames):
        t0 = time.time()
        ret, frame = cap.read()
        
        frame = handle_image(detector, helmet_classifier, jacket_classifier,
                             frame.copy(), output_dir, file_name, save_crop, 
                             save_frame=False, frame_num=frame_num)
          
        writer.write(frame)
        logger.info(f"Frame {frame_num + 1}/{total_frames} - Time: {time.time()- t0}s")
    
    cap.release()
    writer.release()

def main():
    logger.info(f"Parsing arguments ...")

    parser = argparse.ArgumentParser()
    parser.add_argument("--save_crop", action='store_true', help="Flag of whether to save cropped regions or not")
    args = parser.parse_args()

    logger.info(f"Loading components ...")
     
    detector = models["detection"]["rapid"](inference=True)    
    logger.info(f"Loaded detector.")
    helmet_classifier = models["classification"]["efficientnetv2s"](inference=True, type="helmet")    
    logger.info(f"Loaded helmet classifier.")
    jacket_classifier = models["classification"]["efficientnetv2s"](inference=True, type="jacket")    
    logger.info(f"Loaded jacket classifier.")

    logger.info("Inferencing ...")
    for file_name in os.listdir("inputs"):
        input_dir = os.path.join("inputs", file_name)
        output_dir = os.path.join("outputs", file_name)
        if file_name.split(".")[1].lower() == "mp4":
            handle_video(detector, helmet_classifier, jacket_classifier, 
                    input_dir, output_dir, file_name, args.save_crop)
        else:
            handle_image(detector, helmet_classifier, jacket_classifier,
                    cv2.imread(input_dir), output_dir, file_name, args.save_crop)
        logger.info(f"================= Finish {file_name} =================")
    cv2.destroyAllWindows()
    logger.info("Finish inferencing !")

if __name__ == '__main__':
    main()
