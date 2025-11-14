# Automated Object Measurement with a Total Station Using Instance Segmentation
This project accompanies the master's thesis *"Automated Object Measurement with a Total Station Using Instance Segmentation"*.

[![Video ansehen](https://img.youtube.com/vi/xHrBriSdfJM/0.jpg)](https://www.youtube.com/watch?v=xHrBriSdfJM)


## Repository Contents

This repository contains four Python scripts:

- **main_total_station_control.py**  
  Controls the total station via GeoCOM commands. Performs measurements, controls the camera, and enables automated target point acquisition.

- **Object_Segmentation_and_Canny_Edge_Detection_App.py**  
  Performs object detection and segmentation using YOLOv8, followed by Canny edge detection on the segmented areas. Includes a GUI for parameter adjustment.

- **Hough_Line_Transform_App.py**  
  Detects lines using the Hough Transform on pre-segmented objects. Parameters can be adjusted and results visualized via a Tkinter GUI.

- **Harris_Corner_detection_and_DBSCAN_Clustering_App.py**  
  Harris Corner Detection combined with DBSCAN clustering for precise corner detection. GUI allows real-time parameter tuning and visualization of results.

## Installation

1. Install Python 3.10 or later.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
