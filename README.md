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
Inventory: https://www.ncei.noaa.gov/pub/data/ghcn/daily/ghcnd-inventory.txt
Stations: https://www.ncei.noaa.gov/pub/data/ghcn/daily/ghcnd-stations.txt 
Countries: https://www.ncei.noaa.gov/pub/data/ghcn/daily/ghcnd-countries.txt 
States: https://www.ncei.noaa.gov/pub/data/ghcn/daily/ghcnd-states.txt
State_raw: https://raw.githubusercontent.com/georgique/world-geojson/develop/states/usa/new_hampshire.json 



### Data Distribution


### Analysis



### Webapp
Webapp with options to live stream or upload video for real-time object detection

Run below command from project root dir
```sh
uvicorn app.app:app --reload
```

![image](docs/Webapp.png)

---

### References


