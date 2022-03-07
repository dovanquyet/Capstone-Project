import cv2
import numpy as np
from config import config

def crop_bbox(frame, detected_object, expand_ratio=0.08):
    bbox = detected_object.bbox
    frame_h, frame_w = frame.shape[:2]
    cx, cy = frame_w / 2, frame_h / 2
    im = frame.copy()
    x, y, w, h, angle = bbox
    # Translation ===============================================================
    M1 = np.float32([
                [1, 0, cx-x],
                [0, 1, cy-y]
             ])
    im = cv2.warpAffine(im, M1, (frame_w, frame_h))
    # Rotation ==================================================================
    angle = angle%180
    angle = angle-180 if angle >= 90 else angle
    angle = angle-180 if y > frame_h/2 else angle
    M2 = cv2.getRotationMatrix2D(center=(cx, cy), angle=angle, scale=1)
    im = cv2.warpAffine(im, M2, (frame_w, frame_h))
    # Expand bbox to capture larger image =======================================
    x1, x2, y1, y2 = int(cx-w/2), int(cx+w/2), int(cy-h/2), int(cy+h/2)
    x1 = int(max(0, x1 - expand_ratio * w))
    x2 = int(min(frame_w, x2 + expand_ratio * w))
    y1 = int(max(0, y1 - expand_ratio * h))
    y2 = int(min(frame_h, y2 + expand_ratio * h))
    return im[y1:y2, x1:x2]