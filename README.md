# ❄️ BlizzardX – Cold Event Forecasting  
### UMBC DATA606 – Capstone in Data Science  
**Instructor:** Dr. Unal Sakoglu  

**Team Members:**  
- Hema Pushpika Konduru – hemapuk1@umbc.edu  
- Lokeswar Kudumula – lokeswk1@umbc.edu  
- Namruth Goud Thimmapuram – nthimma1@umbc.edu  
- Sree Sai Preetham Kadiyam – FG32258@umbc.edu  

---

## 🌨️ Project Motivation  
Extreme cold events threaten infrastructure, agriculture, and public health. Current forecasts are often broad and slow.  
**BlizzardX** leverages NOAA GHCN-Daily data and machine learning to deliver **precise**, **localized cold event forecasting** with confidence scoring.

---

## 📊 Dataset Overview  
- **Source:** NOAA GHCN-Daily  
- **Focus:** Train on *New Hampshire*, Test on *Vermont*  
- **Timespan:** ~12 years (2010–2022)  
- **Features:** `TMIN`, `TMAX`, `SNOW`, `SNWD`, `PRCP` + temporal features

---

## 🧾 Dataset Details  
The dataset from **ncei.noaa.gov** provides comprehensive environmental data collected globally.  
It includes weather observations (temperature, precipitation, wind speed, humidity), and also oceanographic and geophysical information.  
Long-term records of severe storms, satellite imagery, and Earth trends help support forecasting, climate research, and disaster preparedness.

---

## 🔗 Dataset Access Links  
- **All Data:** [ncei.noaa.gov/data](https://www.ncei.noaa.gov/data/)  
- **Inventory:** [ghcnd-inventory.txt](https://www.ncei.noaa.gov/pub/data/ghcn/daily/ghcnd-inventory.txt)  
- **Stations:** [ghcnd-stations.txt](https://www.ncei.noaa.gov/pub/data/ghcn/daily/ghcnd-stations.txt)  
- **Countries:** [ghcnd-countries.txt](https://www.ncei.noaa.gov/pub/data/ghcn/daily/ghcnd-countries.txt)  
- **States:** [ghcnd-states.txt](https://www.ncei.noaa.gov/pub/data/ghcn/daily/ghcnd-states.txt)  
- **New Hampshire GeoJSON:** [new_hampshire.json](https://raw.githubusercontent.com/georgique/world-geojson/develop/states/usa/new_hampshire.json)
---

## EDA
- Station coverage & Geographic Bias Visual ![image](https://github.com/user-attachments/assets/e18942ae-46ae-4374-a448-f7328196c542)


---

## 🧠 Feature Engineering  
We engineered:  
- Lag features: `TMIN_lag1`, `SNOW_lag1`, etc.  
- Rolling averages  
- Seasonal encodings  
- Interaction terms like `TMIN × SNOW`, `PRCP × SNWD`

---

## ⚙️ Modeling Approach  
- **Prophet:** Forecasts `TMIN` time series per station  
- **XGBoost:** Classifies cold events using a 10th percentile threshold  
- **Output:** Forecast + Binary label + Confidence score

---

## 📈 Evaluation  
- **Regression:** MAE ~ 0.4°C, RMSE < 1°C  
- **Classification:** F1 Score > 96%  
- Generalized well across states (NH → VT)

---

## 📉 Key Visualizations  
- Predicted vs Actual TMIN (Time Series Line Chart)  
- Temperature Distribution (Bar Chart)  
- Prediction Accuracy (Scatter Plot)  
- Daily Temperature Error (Area Chart)  
- Station-wise RMSE Comparison (Bar Chart)

---

## ✅ Conclusion & Impact  
BlizzardX provides **real-time, interpretable**, and **localized cold event detection**.  
It bridges raw climate data with actionable insights for emergency planning, agriculture, and infrastructure management.

---

## 🔗 GitHub & Contact  
- [View Project on GitHub](https://github.com/your-username/blizzardx)

---

## 📽️ Demo  
![image](https://github.com/user-attachments/assets/d33f905b-162c-470e-b81c-748f6a5faa0d)

---

## 📚 References  
- NOAA GHCN-Daily: [https://www.ncei.noaa.gov/data/](https://www.ncei.noaa.gov/data/)  
- Facebook Prophet: [https://facebook.github.io/prophet/](https://facebook.github.io/prophet/)  
- XGBoost: [https://xgboost.ai/](https://xgboost.ai/)  
