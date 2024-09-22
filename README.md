

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

Todo:
- [ ] Think about camera distortion
- [ ] Run through a variety of videos and save field images
- [ ] Label robot bumpers in field images
- [ ] Finetune bumper detection model
- [ ] Solve camera distortion problem
- [ ] Get adequately smooth and accurate robot positions
- [ ] Either a simple algorithm, kalman filter, or SAITS for imputation of missing robot positions
- [ ] Collect time series of robot positions
- [ ] Label key events (scoring, defense, intake)
- [ ] Train a model to predict key events
- [ ] Restructure to run distributed
- [ ] Run on all of 2024 season
- [ ] Train 2023 field keypoint detection model (as second class in the same model) to hopefully improve performance with minimal data for 2025 season