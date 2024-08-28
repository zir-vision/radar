import ultralytics
import cv2
from ultralytics.engine.results import Results
import supervision as sv
import numpy as np
from robots import RobotDetector, RobotRecognizer, VideoTracker
import pickle
from rich.progress import track
import perspect
from utils import kps_ultralytics_to_detection, draw_matches, pad_vertices
import crescendo
field_detector = ultralytics.YOLO("train30-v7-640.pt")





# with open("dets.pkl", "rb") as f:
#     detss = pickle.load(f)

# frames = []

# for i, (frame, dets) in enumerate(detss):
#     # Only keep detections in the top half of the image
#     frame_height = frame.shape[0]
#     height_filter = dets.xyxy[:, 1] < frame_height / 2
#     dets.xyxy = dets.xyxy[height_filter]
#     dets.confidence = dets.confidence[height_filter]

#     filter = dets.confidence > 0.95
#     dets.xy = dets.xyxy[filter][np.newaxis]
#     dets.confidence = dets.confidence[filter][np.newaxis]
#     detected_robots = VideoTracker.get_crops(dets, frame)
#     images = [x[0] for x in detected_robots]
#     frames.append(images)
    
# with open("frames_images.pkl", "wb") as f:
#     pickle.dump(frames, f)

# with open("frames_images.pkl", "rb") as f:
#     frames = pickle.load(f)

# robot_recognizer = RobotRecognizer()
# embeddings = []

# for frame in track(frames):
#     embeddings.append(list(robot_recognizer.recognize(frame)))

# with open("embeddings.pkl", "wb") as f:
#     pickle.dump(embeddings, f)

# with open("embeddings.pkl", "rb") as f:
#     embeddings = pickle.load(f)

# robot_recognizer = RobotRecognizer()

# frames = robot_recognizer.cluster(embeddings)

# with open("frames.pkl", "wb") as f:
#     pickle.dump(frames, f)

# robot_detector = RobotDetector("robot_detect.pt")
cap = cv2.VideoCapture("sac-low-fps.mp4")

i = -1
# out_size = (1920, 1400)
out_size = (1920, 1080)
# writer = cv2.VideoWriter("kp.avi", cv2.VideoWriter_fourcc(*"MJPG"), cap.get(cv2.CAP_PROP_FPS), (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
writer = cv2.VideoWriter("perspect.avi", cv2.VideoWriter_fourcc(*"MJPG"), cap.get(cv2.CAP_PROP_FPS), out_size)
SCALE = 200
# detss = []

persp: perspect.VideoPerspective | None = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    i += 1
    if i < 300:
        continue
    
    frame = cv2.resize(frame, (640, 640))

    results: Results = field_detector(frame)[0]
    
    kps = sv.KeyPoints.from_ultralytics(results)

    dets = kps_ultralytics_to_detection(results)

    biggest_det = max(dets, key=lambda x: (x.x2 - x.x1) * (x.y2 - x.y1))
    # filter = kps.confidence > 0.50
    # kps.xy = kps.xy[filter][np.newaxis]
    # kps.confidence = kps.confidence[filter][np.newaxis]

    print(f"{kps.xy.shape=}")
    print(f"{kps.confidence.shape=}")
    # vertex_annotator = sv.VertexAnnotator(radius=4)
    # annotated_image = vertex_annotator.annotate(frame.copy(), kps)

    if persp is None:
        persp = perspect.VideoPerspective(biggest_det, kp_threshold=0.8, padding=[0, 200, 200, 200], scale=SCALE)
    else:
        persp.update(biggest_det)
    # print(f"{m=}")
    warped = persp.warp_image(frame, True)
    cv2.imwrite("warped.jpg", warped)

    # matches_img = draw_matches(frame, biggest_det.keypoints, scale=SCALE)
    # cv2.imwrite("matches.jpg", matches_img)
    
    # Stack images vertically
    # collage = np.zeros((warped.shape[0] + matches_img.shape[0], max(warped.shape[1], matches_img.shape[1]), 3), np.uint8)
    # collage[:warped.shape[0], :warped.shape[1]] = warped
    # collage[warped.shape[0]:, :matches_img.shape[1]] = matches_img

    writer.write(cv2.resize(warped, out_size))

#     dets = robot_detector.detect(frame)

#     detss.append((frame, dets))

#     # writer.write(anntd)

    if i > 1000:
        writer.release()
        # with open("dets.pkl", "wb") as f:
        #     pickle.dump(detss, f)
        break
    

cap.release()