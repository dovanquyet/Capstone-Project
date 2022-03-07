import numpy as np

class DetectedObject:

    def __init__(self, bbox, confidence):
        # Bounding box
        self.bbox = bbox
        self.confidence = confidence
        # Centroid and corners
        self.corners = self._get_corners()
        # Object information - classification results
    
    def _get_corners(self, expand_ratio=0):
        cx, cy, w, h, angle = self.bbox
        self.centroid = (int(cx), int(cy))
        angle = angle / 180 * np.pi
        w_e, h_e = w * (1 + expand_ratio), h * (1 + expand_ratio)
        c, s = np.cos(angle), np.sin(angle)
        R = np.asarray([[c, s], [-s, c]])
        pts = np.asarray([[-w_e/2, -h_e/2], [w_e/2, -h_e/2], [w_e/2, h_e/2], [-w_e/2, h_e/2]])
        corners = [([cx, cy] + pt @ R).astype(int) for pt in pts]
        return corners
