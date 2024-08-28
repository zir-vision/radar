import cv2
import crescendo
import supervision
import numpy as np
from utils import Detection, plot_field, KP

class Perspective:
    @staticmethod
    def find_homography(kps: Detection, kp_threshold: float = 0.8, padding: tuple[int, int, int, int] = [0, 200, 200, 200], scale: int = 200):
        points = []
        mask = []
        for kp in kps.keypoints:
            if kp.confidence > kp_threshold:
                mask.append(True)
                points.append((kp.x, kp.y))
            else:
                mask.append(False)
        source = np.array(points, dtype=np.float32)
        print(f"{source=}")
        target = np.array(crescendo.VERTICES, dtype=np.float32)[mask] * scale
        # Apply padding
        target[:, 0] += padding[3]
        target[:, 1] += padding[0]
        print(f"{target=}")
        matrix, _ = cv2.findHomography(source, target)
        return matrix
    def __init__(self, kps: Detection, kp_threshold: float = 0.8, padding: tuple[int, int, int, int] = [0, 200, 200, 200], scale: int = 200):
        """
        Finds the homography matrix to transform points from the source to the destination.
        Padding is the the shape of [top, right, bottom, left].
        """
        self.kps = kps
        self.kp_threshold = kp_threshold
        self.padding = padding
        self.scale = scale
        self.matrix = self.find_homography(kps, kp_threshold, padding, scale)

    def warp_points(self, points: list[tuple[float, float]]) -> list[tuple[float, float]]:
        """
        Warps a list of points using the homography matrix.
        """
        points = np.array(points, dtype=np.float32)
        warped = cv2.perspectiveTransform(points.reshape(1, -1, 2), self.matrix)
        return warped.reshape(-1, 2)

    def warp_image(self, image, annotate=False):
        """
        Warps an image using a homography matrix.
        """
        


        warped = cv2.warpPerspective(image, self.matrix, (int(crescendo.FIELD_WIDTH*self.scale+(self.padding[1]*2)), int(crescendo.FIELD_HEIGHT*self.scale+(self.padding[2]*2))))
        if annotate:
            # Warp the keypoints
            kps = self.kps.keypoints
            warped_points = self.warp_points([(kp.x, kp.y) for kp in kps])
            warped_kps = []
            for i, kp in enumerate(kps):
                x, y = warped_points[i]
                warped_kps.append(KP(int(x), int(y), kp.confidence))
            warped = plot_field(warped, warped_kps)
        return warped
    

class VideoPerspective(Perspective):
    """
    A smoothened perspective transformation for videos.
    Uses a simple rolling average to smoothen the transformation.
    """
    
    def update(self, kps: Detection):
        """
        Updates the homography matrix with new keypoints.
        """
        self.kps = kps
        try:
            prev_matrices = self.prev_matrices
        except AttributeError:
            prev_matrices = []
            self.prev_matrices = prev_matrices

        matrix = self.find_homography(kps, self.kp_threshold, self.padding, self.scale)
        prev_matrices.append(matrix)
        if len(prev_matrices) > 25:
            prev_matrices.pop(0)
        self.matrix = np.mean(prev_matrices, axis=0)
        