# BEV_Tracker
# BEV Tracker: Multi-View Motion Trajectory Mapping Tool

BEV Tracker is a lightweight real-time software system for accurate single-person motion trajectory mapping using multi-view Kinect cameras in indoor environments. This tool addresses challenges such as occlusion and reflection, which often lead to erroneous detections, by incorporating advanced post-processing strategies.

## Key Features

- **Real-Time Motion Tracking**: Tracks the motion trajectory of an individual in real-time using multi-view Kinect cameras.
- **Momentum Trajectory Smoothing**: Uses an enhanced Kalman filter with historical velocity weighting to smooth motion trajectories and improve temporal consistency.
- **Prior-Knowledge-Driven Velocity Constraint**: Prevents unrealistic jumps in the trajectory by imposing limits based on human motion priors.
- **BEV (Bird’s-Eye View) Representation**: Projects 2D detections onto the BEV plane through geometric transformations, ensuring high compatibility with a variety of sensors.
- **Scalable and Extendable**: Designed to be scalable for multi-person tracking and can be extended to include depth-aware enhancements in the future.

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/qawozeixructyv/BEV_Tracker.git
   cd BEV_Tracker
