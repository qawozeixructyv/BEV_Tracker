# BEV Tracker: Multi-View Motion Trajectory Mapping Tool

BEV Tracker is a lightweight real-time software system for accurate single-person motion trajectory mapping using multi-view Kinect cameras in indoor environments. This tool addresses challenges such as occlusion and reflection, which often lead to erroneous detections, by incorporating advanced post-processing strategies.
![image](https://github.com/user-attachments/assets/93a90a2b-a1e6-417f-93a5-a5221a5c8c69)

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
```
2.Install required dependencies:
```bash
   pip install -r requirements.txt
```
   
3.Set up the environment and run the system on multi-view Kinect cameras.

## Usage
1.Start the system:
The software can be run using the following command:
```bash
   python bev_tracking_gui.py
```
   
2.Configure Kinect Cameras:
Follow the on-screen instructions to set up multiple Kinect cameras in a synchronized master-slave configuration. Each camera’s calibration will be required to enable real-time video feeds.

3.Visualize Trajectories:
The real-time motion trajectory of the detected subject will be projected onto the Bird's-Eye View (BEV) plane and displayed in the system’s UI.
