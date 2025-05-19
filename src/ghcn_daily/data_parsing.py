import requests
import pandas as pd

class DataParser:
    @staticmethod
    def parse_stations(url):
        r = requests.get(url)
        lines = r.text.split("\n")
        data = []
        for line in lines:
            if line:
                data.append({
                    "ID": line[0:11].strip(),
                    "LATITUDE": float(line[12:20].strip()),
                    "LONGITUDE": float(line[21:30].strip()),
                    "ELEVATION": float(line[31:37].strip()),
                    "STATE": line[38:40].strip(),
                    "NAME": line[38:68].strip(),
                })
        return pd.DataFrame(data)

    @staticmethod
    def parse_inventory(url):
        r = requests.get(url)
        lines = r.text.split("\n")
        data = []
        for line in lines:
            if line:
                data.append({
                    "ID": line[0:11].strip(),
                    "LATITUDE": float(line[12:20].strip()),
                    "LONGITUDE": float(line[21:30].strip()),
                    "ELEMENT": line[31:35].strip(),
                    "FIRSTYEAR": int(line[36:40].strip()),
                    "LASTYEAR": int(line[41:45].strip())
                })
        return pd.DataFrame(data)

    @staticmethod
    def parse_countries(url):
        r = requests.get(url)
        lines = r.text.split("\n")
        data = []
        for line in lines:
            if line:
                data.append({
                    "CODE": line[0:2].strip(),
                    "NAME": line[3:64].strip()
                })
        return pd.DataFrame(data)

    @staticmethod
    def parse_states(url):
        r = requests.get(url)
        lines = r.text.split("\n")
        data = []
        for line in lines:
            if line:
                data.append({
                    "CODE": line[0:2].strip(),
                    "NAME": line[3:50].strip()
                })
        return pd.DataFrame(data)
    
    @staticmethod
    def parse_data_dly(line):
        data = []
        for i in range(21, 269, 8):
            value = int(line[i:i+5])
            mflag = line[i+5]
            qflag = line[i+6]
            sflag = line[i+7]
            data.extend([value, mflag, qflag, sflag])
        return {
            "ID": line[0:11].strip(),
            "YEAR": int(line[11:15]),
            "Month": int(line[15:17]),
            "ELEMENT": line[17:21].strip(),
            "DATA": data
        }

