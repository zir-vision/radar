import ultralytics
import cv2
from ultralytics.engine.results import Results
import supervision as sv
import perspect
from utils import kps_ultralytics_to_detection, iter_video
field_detector = ultralytics.YOLO("train30-v7-640.pt")

cap = cv2.VideoCapture("sac-low-fps.mp4")

out_size = (1920, 1080)
SCALE = 1


def track_robots_in_video(video_path: str, output_path: str = "output.avi"):
    persp: perspect.VideoPerspective = perspect.VideoPerspective()
    
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"MJPG"), cap.get(cv2.CAP_PROP_FPS), out_size)

    for frame in iter_video(video_path):
        frame = cv2.resize(frame, (640, 640))

        results: Results = field_detector(frame)[0]
        
        kps = sv.KeyPoints.from_ultralytics(results)

        dets = kps_ultralytics_to_detection(results)

        biggest_det = max(dets, key=lambda x: (x.x2 - x.x1) * (x.y2 - x.y1))
        print(f"{kps.xy.shape=}")
        print(f"{kps.confidence.shape=}")

        persp.update(biggest_det)

        warped = persp.warp_image(frame)
        
    writer.release()
if __name__ == "__main__":
    track_robots_in_video("sac-low-fps.mp4")
    