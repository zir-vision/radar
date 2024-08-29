import cv2
import crescendo
from dataclasses import dataclass
from ultralytics.engine.results import Results


@dataclass
class KP:
    x: float
    y: float
    confidence: float

@dataclass
class Detection:
    class_index: int
    x1: float
    y1: float
    x2: float
    y2: float
    keypoints: list[KP]


def plot_field(image, keypoints: list[KP]):
    """
    Plots the keypoints on the field image.
    """
    # image = cv2.resize(image, (image.shape[1]*4, image.shape[0]*4))
    for i, kp in enumerate(keypoints):
        if kp.confidence < 0.5:
            continue
        print(f"Point {i}: {kp.x}, {kp.y}")
        cv2.circle(image, (kp.x, kp.y), 5, (0, 255, 0), -1)
        cv2.putText(image, crescendo.KEYPOINT_MAP[i], (kp.x, kp.y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    for edge in crescendo.EDGES:
        # Edge is a tuple of two keypoint indices
        kp1 = keypoints[edge[0]]
        kp2 = keypoints[edge[1]]
        if kp1.confidence < 0.5 or kp2.confidence < 0.5:
            continue
        cv2.line(image, (kp1.x, kp1.y), (kp2.x, kp2.y), (255, 0, 0), 2)

    return image

def plot_yolov8_kp(image, label_file_lines: list[str]):
    """
    Plots the keypoints from a YOLOv8 pose dataset on an image.
    """
    image = cv2.resize(image, (image.shape[1]*4, image.shape[0]*4))
    height, width, _ = image.shape
    # print(height, width)
    for line in label_file_lines:
        parts = line.split(" ")
        keypoints: list[tuple[float, float, int]] = [(float(parts[i]), float(parts[i + 1]), int(parts[i + 2])) for i in range(5, len(parts), 3)]
        print(keypoints)
        kps = []
        for i, (px, py, visibility) in enumerate(keypoints):
            x = int(px * width)
            y = int(py * height)
            # print(f"Point {i}: {x}, {y}")
            if visibility != 0:
                kps.append(KP(x, y, 1.0))
            else:
                kps.append(KP(x, y, 0.0))
        plot_field(image, kps)
    return image

def plot_yolov8_kp_from_file(img_path, label_path, output_path):
    with open(label_path, "r") as f:
        lines = f.readlines()
    image = cv2.imread(img_path)
    annotated_image = plot_yolov8_kp(image, lines)
    cv2.imwrite(output_path, annotated_image)

def kps_ultralytics_to_detection(results: Results) -> list[Detection]:
    kps = results.keypoints.numpy().data
    # print(f"{kps.shape=}")
    detections = []
    for i in range(len(results.boxes)):
        keypoints = []
        for kp in kps[i].reshape(-1, 3):
            # print(f"{kp.shape=}")
            keypoints.append(KP(kp[0], kp[1], kp[2]))
        box = results.boxes.xyxy[i]
        clazz = results.boxes.cls[i]
        detections.append(Detection(clazz, box[0], box[1], box[2], box[3], keypoints))
    return detections



def draw_matches(image, kps: list[KP], scale=200):
    print(f"{image.shape=}")
    import numpy as np
    mask = [(True if kp.confidence > 0.5 else False) for kp in kps]
    kpsl = [(kp.x, kp.y) for kp in kps]
    kps_arr = np.array(kpsl, dtype=np.float32)[mask]
    target_img = crescendo.draw_2d_field_ex(scale)
    
    # Create image with both images side by side
    matched = np.zeros((max(image.shape[0], target_img.shape[0]), image.shape[1] + target_img.shape[1], 3), np.uint8)
    matched[:image.shape[0], :image.shape[1]] = image
    matched[:target_img.shape[0], image.shape[1]:] = target_img

    # Offsets for second image
    offset_x = image.shape[1]

    # Draw lines between keypoints
    for i, kp in enumerate(kps_arr):
        if mask[i]:
            print(f"Drawing line from {kp} to {crescendo.VERTICES[i]}")
            cv2.line(matched, (int(kp[0]), int(kp[1])), (int(crescendo.VERTICES[i][0]*scale + offset_x), int(crescendo.VERTICES[i][1]*scale)), (0, 255, 0), 2)


    return matched


def iter_video(video_path, start_frame: int = 0, end_frame: int | None = None):
    """
    Iterates over the frames of a video.
    """
    cap = cv2.VideoCapture(video_path)
    i = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if i < start_frame:
            # print(f"Skipping frame {i}")
            i += 1
            continue
        if end_frame is not None and i > end_frame:
            # print(f"Reached end frame {end_frame}")
            break
        yield frame
        i += 1
    cap.release()

if __name__ == "__main__":
    import sys
    plot_yolov8_kp_from_file(sys.argv[1], sys.argv[2], sys.argv[3])