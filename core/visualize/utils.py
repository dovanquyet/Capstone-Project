import cv2
import numpy as np
from .colors import *

def draw_inferences(frame, inferences):
    for inference in inferences:
        detected_object, helmet_label, jacket_label = inference
        contours = np.array(detected_object.corners)
        color = colors[(helmet_label, jacket_label)]
        cv2.polylines(frame, [contours], isClosed=True, color=color, thickness=3, lineType=cv2.LINE_4)  
    return frame