import pickle
import ultralytics
import cv2
from ultralytics.engine.results import Results
import perspect
from utils import kps_ultralytics_to_detection, iter_video, Box
import numpy as np
import numpy.typing as npt
from robots import RobotDetector, RobotRecognizer
from rich.pretty import pprint
out_size = (1920, 1080)
SCALE = 1
PADDING = (0, 200, 200, 200)

def track_robots_in_video(video_path: str, output_path: str = "output.avi", intermediate_path: str = "intermediate.pickle"):
    field_detector = ultralytics.YOLO("train30-v7-640.pt")
    persp: perspect.VideoPerspective = perspect.VideoPerspective(perspect.PerspectiveConfig(scale=SCALE, padding=PADDING))

    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"MJPG"), 15, out_size)

    robot_detector = RobotDetector("robot_detect.pt")
    robot_recognizer = RobotRecognizer()

    # List of each frame
    # Each frame is a list of robot detections
    # Each detection is a tuple of a box and the embeddings
    data: list[list[tuple[Box, npt.NDArray]]] = []

    for frame in iter_video(video_path, 100):
        frame = cv2.resize(frame, (640, 640))
        cv2.imwrite("frame.jpg", frame)

        results: Results = field_detector(frame, conf=0.95)[0]

        dets = kps_ultralytics_to_detection(results)

        if len(dets) == 0:
            data.append(None)
            continue

        biggest_det = max(dets, key=lambda x: x.box.area())
        
        persp.update(biggest_det)

        warped = persp.warp_image(frame)

        robot_dets = robot_detector.detect(warped)

        if len(robot_dets) == 0:
            data.append([])
            continue
        cropped_robot_imgs = []
        for robot_det in robot_dets:
            box = robot_det.box
            cropped = warped[int(box.y1) : int(box.y2), int(box.x1) : int(box.x2)]
            cropped_robot_imgs.append(cropped)

        embeddings = robot_recognizer.recognize(cropped_robot_imgs)
        data.append(list(zip(map(lambda x: x.box, robot_dets), embeddings)))
        # break
    with open(intermediate_path, "wb") as f:
        pickle.dump(data, f)
    pprint(data)
    writer.release()


if __name__ == "__main__":
    track_robots_in_video("sac-low-fps.mp4")
