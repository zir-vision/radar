from ultralytics import YOLO
from ultralytics.engine.results import Results
import cv2
import supervision as sv
import numpy as np
from typing import Iterable
from dataclasses import dataclass
from transformers import Dinov2Model, AutoModel, AutoImageProcessor
from umap import UMAP
from hdbscan.flat import HDBSCAN_flat
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.preprocessing import normalize
from utils import det_ultralytics_to_detection, Detection
class RobotDetector:
    def __init__(self, model_path: str):
        self.model = YOLO(model_path)
        self.det_size = (640, 640)

    def detect(self, img) -> list[Detection]:
        original_size = img.shape[:2]
        resize_ratio = (
            original_size[1] / self.det_size[0],
            original_size[0] / self.det_size[1],
        )
        det_img = cv2.resize(img, self.det_size)
        results: Results = self.model.predict(det_img, conf=0.5)[0]

        dets = det_ultralytics_to_detection(results)
        # Adjust the detections to the original image size
        # dets.xyxy = dets.xyxy * np.array(
        #     [resize_ratio[0], resize_ratio[1], resize_ratio[0], resize_ratio[1]]
        # )
        for det in dets:
            det.box.x1 *= resize_ratio[0]
            det.box.x2 *= resize_ratio[0]
            det.box.y1 *= resize_ratio[1]
            det.box.y2 *= resize_ratio[1]
        
        # box_annotator = sv.BoxAnnotator()
        # annotated_image = box_annotator.annotate(img.copy(), dets)
        return dets

    def detect_video(self, video: Iterable[np.ndarray]):
        for frame in video:
            yield self.detect(frame)

class RobotRecognizer:
    model: Dinov2Model

    def __init__(self, model_name: str = "facebook/dinov2-base"):
        self.preprocessor = AutoImageProcessor.from_pretrained(model_name)
        self.model = Dinov2Model.from_pretrained(model_name)

    def recognize(self, imgs: Iterable[np.ndarray]) -> np.ndarray:
        
        inputs = self.preprocessor(images=list(imgs), return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
        embeddings = outputs.pooler_output.detach().numpy()
        print(f"{embeddings.shape=}")
        return embeddings

    def recognize_video(self, video: Iterable[Iterable[np.ndarray]]):
        for frame in video:
            yield self.recognize(frame)

    def cluster(
        self,
        embeddings: list[list[np.ndarray]],
        robots_on_field: int = 6,
        cluster_dim: int = 2,
        min_robot_cluster_size: int | None = None,
        debug: bool = False,
    ) -> list[list[tuple[int]]]:
        if min_robot_cluster_size is None:
            min_robot_cluster_size = round((len(embeddings) / robots_on_field) * 0.75)

        umap = UMAP(n_components=cluster_dim, min_dist=0.0)

        flattened_embeddings = []
        labels = []

        for i, frame in enumerate(embeddings):
            for j, emb in enumerate(frame):
                # print(emb.shape)
                flattened_embeddings.append(emb[0])
                labels.append((i, j))

        embeddings_array = np.array(flattened_embeddings)
        embeddings_array = normalize(embeddings_array)
        u = umap.fit_transform(flattened_embeddings)
        if debug:
            plt.scatter(u[:, 0], u[:, 1]).get_figure().savefig("umap.png")
            plt.clf()
        
        robot_clusterer = HDBSCAN_flat(u, n_clusters=robots_on_field, max_cluster_size=len(embeddings)/robots_on_field)
        if debug:
            print(f"{robot_clusterer.labels_.max()+1} clusters found")
            robot_color_palette = sns.color_palette('Paired', robot_clusterer.labels_.max() + 1)
            robot_cluster_colors = [robot_color_palette[x] if x >= 0
                            else (0.5, 0.5, 0.5)
                            for x in robot_clusterer.labels_]
            # robot_cluster_member_colors = [sns.desaturate(x, p) for x, p in
            #                         zip(robot_cluster_colors, robot_clusterer.probabilities_)]
            robot_cluster_member_colors = robot_cluster_colors
            plt.scatter(u[:, 0], u[:, 1], c=robot_cluster_member_colors).get_figure().savefig("robot_clusters.png")
            plt.clf()
        dict_frames = [{} for _ in range(len(embeddings))]

        for i, label in enumerate(labels):
            dict_frames[label[0]][label[1]] = robot_clusterer.labels_[i]

        frames = []
        for d in dict_frames:
            items = sorted(d.items(), key=lambda x: x[0])
            frames.append([x[1] for x in items])


        return frames
        

class VideoTracker:
    frames: list[list[tuple[np.ndarray, np.ndarray]]]

    def __init__(
        self, robot_detector: RobotDetector, robot_recognizer: RobotRecognizer
    ):
        self.robot_detector = robot_detector
        self.robot_recognizer = robot_recognizer
        self.frames = []

    @staticmethod
    def get_crops(dets: sv.Detections, img: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
        detected_robots = []
        for det in dets.xyxy:
            cropped = img[int(det[1]) : int(det[3]), int(det[0]) : int(det[2])]
            detected_robots.append((cropped, det))
        return detected_robots

    def detect(self, imgs: Iterable[np.ndarray]):
        for img in imgs:
            dets = self.robot_detector.detect(img)
            detected_robots = self.get_crops(dets, img)
            yield detected_robots
