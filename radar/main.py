import ultralytics
import cv2
from ultralytics.engine.results import Results
import supervision as sv
import perspect
from utils import kps_ultralytics_to_detection, iter_video
field_detector = ultralytics.YOLO("train30-v7-640.pt")

cap = cv2.VideoCapture("sac-low-fps.mp4")

out_size = (1920, 1080)
writer = cv2.VideoWriter("perspect.avi", cv2.VideoWriter_fourcc(*"MJPG"), cap.get(cv2.CAP_PROP_FPS), out_size)
SCALE = 200

persp: perspect.VideoPerspective | None = None

for frame in iter_video("sac-low-fps.mp4"):
    
    
    frame = cv2.resize(frame, (640, 640))

    results: Results = field_detector(frame)[0]
    
    kps = sv.KeyPoints.from_ultralytics(results)

    dets = kps_ultralytics_to_detection(results)

    biggest_det = max(dets, key=lambda x: (x.x2 - x.x1) * (x.y2 - x.y1))
    print(f"{kps.xy.shape=}")
    print(f"{kps.confidence.shape=}")

    if persp is None:
        persp = perspect.VideoPerspective(biggest_det, kp_threshold=0.8, padding=[0, 200, 200, 200], scale=SCALE)
    else:
        persp.update(biggest_det)

    warped = persp.warp_image(frame, True)
    cv2.imwrite("warped.jpg", warped)

    writer.write(cv2.resize(warped, out_size))
