# Automated Object Measurement with a Total Station Using Instance Segmentation
*(Geodata Technology M.Eng. Thesis Project — THWS, 2025)*

[![Video ansehen](https://img.youtube.com/vi/xHrBriSdfJM/0.jpg)](https://www.youtube.com/watch?v=xHrBriSdfJM)

---

## Overview

This repository contains the public version of the software prototype developed during my master's thesis in **Geodata Technology (M.Eng.) at Technische Hochschule Würzburg-Schweinfurt (THWS), 2025**.

The project evaluates whether a **Computer Vision (CV) pipeline** can support **automated object measurement with a total station**.
It combines:

* **YOLOv8 instance segmentation**
* **Canny edge detection**
* **Progressive probabilistic Hough line transform**
* **Harris corner detection**
* **DBSCAN clustering**
* **GeoCOM-based device control** of a Leica MS50 (all GeoCOM commands removed in this public version)

⚠️ **Important:**
The original measurement hardware, the trained YOLO model, and the physical object used for testing **are unique and not included**.
Images from other scenes cannot be segmented without retraining a model.

---

## Workflow Overview

The pipeline consists of six main CV steps (a–f) and an optional measurement step:

![Workflow](./assets/pipeline_workflow.jpg)

**(a)** Image Capture
**(b)** Instance segmentation → Masked crop
**(c)** Canny edge detection
**(d)** Hough line transform
**(e)** Harris corner detection → DBSCAN clustering
**(f)** Cluster centers transfer
(Optional) Total station control via GeoCOM (GeoCOM commands not included in this repository, only schematic flowchart)

---

## Repository Structure

```text
project/
│
├── README.md
├── requirements.txt
│
├── assets/
│   ├── computer_vision_pipeline.jpg
│   └── total_station_workflow_.jpg
│
├── segmentation_and_canny/
│   └── Object_Segmentation_and_Canny_Edge_Detection_App.py        # (a)-(c) YOLOv8 segmentation + interactive Canny
│
├── hough_transform/
│   └── Hough_Line_Transform_App.py                                # (d) Progressive probabilistic Hough transform
│
├── harris_corner_detection/
│   └── Harris_Corner_detection_and_DBSCAN_Clustering_App.py       # (e) Harris + DBSCAN corner clustering
│
└── total_station_control/
    └── main_total_station_control.py                              # (d) Structure for total-station workflow (GeoCOM removed)
```

---

## Scripts and Their Role in the Workflow

### **1. `Object_Segmentation_and_Canny_Edge_Detection_App.py` (Steps a–c)**

* Runs YOLOv8 instance segmentation
* Crops the detected object (in this workflow the used steel beam)
* Opens an interactive Canny GUI with manual + automatic methods
* Saves cropped object and edges

### **2. `Hough_Line_Transform_App.py` (Step d)**

* Interactive GUI for the Progressive probabilistic Hough transform
* Tunable parameters + live preview of detected lines
* Saves a line-only image on black background

### **3. `Harris_Corner_detection_and_DBSCAN_Clustering_App.py` (Step e)**

* Harris corner detection + DBSCAN clustering
* Tunable parameters + live preview of detected corner clusters
* Saves: visual output and list of final cluster centers (clustered object corners)

### **4. `main_total_station_control.py` (Step f, optional)**

* Public, cleaned structure of the automated total-station workflow
* All **GeoCOM commands removed** due to licensing and confidentiality
* Keeps the architecture to allow users to integrate their own device control
* Workflow:
  * image capture → segmentation → edges → lines → corners → (optionally) aim at corner

---

## Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## License

MIT License — see `LICENSE`.

---
