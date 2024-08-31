

Process:
- Load models
- Open video
- For each frame
    - Run field keypoint detection
    - Find biggest field
    - Undistort
    - Run robot detection
    - For each robot
        - Find embedding
        - Record frame number, position, and embedding

- Intermediate save (pickle)
- Run umap (dimensionality reduction)
- Run dbscan (clustering)
- Create list of frames with robot positions
- (Optional) Make video with robot positions