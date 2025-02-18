# BlizzardX
BlizzardX: Cracking the Code of Snow &amp; Temperature Swings


## 👥 Team Members
- **Team H**    
  - [Hema Pushpika Konduru]
  - [Lokeswar Kudumula] 
  - [Namruth Goud Thimmapuram]
  - [Sree Sai Preetham Kadiyam] 

---

## 📌 Project Overview
**Research Question**:  
How do snowfall and snow depth influence temperature fluctuations in New Hampshire? Can machine learning models predict temperature changes and extreme cold events with high accuracy?  

**Objective**:  
This project analyzes historical weather data from NOAA GHCN-Daily to:  
1. Investigate correlations between snowfall, snow depth, and temperature extremes.  
2. Build regression models to predict temperature.  
3. Classify days with extreme cold events.  

---

## 📂 Dataset
**Source**: NOAA Global Historical Climatology Network Daily (GHCN-Daily)  
- [Station Data](https://www.ncei.noaa.gov/pub/data/ghcn/daily/ghcnd-stations.txt)  
- [Daily Records](https://www.ncei.noaa.gov/pub/data/ghcn/daily/all/)  
- **Filter**: Focused on **New Hampshire stations only** (11 stations).  

**Key Attributes**:  
- `SNOW` (snowfall),
- `SNWD` (snow depth),
- `TMAX` (Maximum temperature (tenths of degrees C)), 
- `TMIN` (Minimum temperature (tenths of degrees C)),
- `PRCP`(Precipitation (tenths of mm)), 
- `WT04`(Ice pellets, sleet, snow pellets, or small hail), 
- `WT09` (Blowing or drifting snow), 
- `WT18`(Snow, snow pellets, snow grains, or ice crystals).  

---

