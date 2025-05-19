<h1 align="center">BlizzardX</h1>
<h3 align="center">UMBC DATA606 - Capstone in Data Science</h3>
<h4 align="right">Dr. Unal Sakoglu</h4>
<h4 align="left">Hema Pushpika Konduru (hemapuk1@umbc.edu)</h4>
<h4 align="left">Lokeswar Kudumula (lokeswk1@umbc.edu)</h4>
<h4 align="left">Namruth Goud Thimmapuram (nthimma1@umbc.edu)</h4>
<h4 align="left">Sree Sai Preetham Kadiyam (FG32258@umbc.edu)</h4>
---

<p align="center">
  <img width="900" height="" src="docs/demo.gif">
</p>


### Dataset details
The dataset from <b>ncei.noaa.gov</b> provides comprehensive environmental data collected from across the globe. It includes weather observations such as temperature, precipitation, wind speed, and humidity on hourly, daily, and monthly scales. The site also hosts long-term climate records, including trends, normals, and extremes. Oceanographic data like sea surface temperature, tides, salinity, and wave height are available, along with geophysical information on earthquakes, tsunamis, and Earth's magnetic field. Additionally, satellite imagery offers insights into cloud cover, land surface changes, and ocean conditions. The platform also includes detailed records of severe storms such as hurricanes and tornadoes. This rich dataset supports climate research, weather forecasting, disaster preparedness, and environmental monitoring.

**Size:**
- BDD100k: 7GB
- Polish12k: 11GB

**Dataset link:**
All Data - https://www.ncei.noaa.gov/data/

**Image Resolution:**
- BDD100K
![image](docs/bdd100k.png)
- Polish12K
![image](docs/polish12k.png)
### Data Distribution
The dataset contain 11 classes:
- Car (Vehicles without a trailer)
- Different-Traffic-Sign 
- Green-Traffic-Light
- Motorcycle
- Pedestrian (People and cyclists)
- Pedestrian-Crossing (Pedestrian crossings)
- Prohibition-Sign (All prohibition signs)
- Red-Traffic-Light (Red traffic lights for cars only
- Pedestrians are not annotated)
- Speed-Limit-Sign (Speed limit signs)
- Truck (Vehicles with a trailer)
- Warning-Sign (Warning signs)

### Analysis
It is an image classification problem. When a new traffic image is passed as input it should detect all kinds of traffic signals, vehicles, pedestrians etc.

### Custom YOLOv8n Model
YOLOv8 Nano is optimized for speed and resource efficiency at the cost of some accuracy.

**Model Training Workflow:**

![image](docs/Workflow.png)

**Bootstrap Setup:**
- `bootstrap.sh` initializes the project environment and directories.

**Model Paths and Training:**
- `config.yaml` Manages paths for images, labels, and models.

**Data Configuration:**
- `data.yaml`: Defines training (13 classes: pedestrian, car, traffic lights, etc.).
- `data_finetune.yaml`: Focuses on fine-tuning (11 classes: traffic signs, pedestrians).

**Utility Functions:**
- `utilities.py` provides helper functions for data handling and training

**Model Architecture:**

![image](docs/YOLOv8-architecture.png)

**Data Preprocessing:**
- Label Conversion Process
  - BDD100K to COCO Format
  - COCO to YOLO Format

![image](docs/YOLO%20Label.png)

**Model Training:**
- **Training:**
  - BDD100K: 100,000 images for object detection in autonomous driving.
- **Finetuning:**
  - Polish12k: 12,000 images for object detection in autonomous driving.

**Training Configuration:**
 - Epochs: 300 (model was trained for 300 iterations)
 - Batch Size: 64 (images processed per batch for each training step)
 - Image Size: 640x640 pixels (standard input size for balanced accuracy and speed)
 - Patience: 100 (training stops if no significant improvement is seen after 100 consecutive epochs)
   
**Device:**
- 2 x Nvidia H100 (80GB) on CUDA

**Cloud:**
- Vast AI (https://vast.ai/)
- Lambda Labs (https://lambdalabs.com/)

### Model Evaluation
- Mean Average Precision (mAP)
- Average Precision at IoU=0.50 (AP50)
- F1 Score

**F1-Confidence Curve**
![image](docs/F1-Confidence%20Curve.png)

**Confusion Matrix**
![image](docs/Confusion%20Matrix.png)


### Webapp
Webapp with options to live stream or upload video for real-time object detection

Run below command from project root dir
```sh
uvicorn app.app:app --reload
```

![image](docs/Webapp.png)

---

### References
- You Only Look Once: Unified, Real-Time Object Detection: https://arxiv.org/abs/1506.02640
- Ultralytics YOLOv8: https://github.com/ultralytics/ultralytics
- YOLOv8 Implementation using Pytorch: https://github.com/jahongir7174/YOLOv8-pt
- BDD100k: https://doc.bdd100k.com/download.html
- Polish12k: https://www.kaggle.com/datasets/mikoajkoek/traffic-road-object-detection-polish-12k


