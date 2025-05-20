import requests
import pandas as pd
import logging

# Configure logging to console only
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

class DataParser:
    def __init__(self):
        self.logger = setup_logging()
        self.logger.info("Initializing DataParser")

    def parse_stations(self, url):

        self.logger.info(f"Fetching and parsing station data from {url}")
        try:
            response = requests.get(url)
            if response.status_code != 200:
                self.logger.error(f"Failed to retrieve data from {url}. Status code: {response.status_code}")
                raise ValueError(f"Failed to retrieve data from {url}. Status code: {response.status_code}")
            
            lines = response.text.split("\n")
            data = []
            for line in lines:
                if line.strip():
                    try:
                        data.append({
                            "ID": line[0:11].strip(),
                            "LATITUDE": float(line[12:20].strip()),
                            "LONGITUDE": float(line[21:30].strip()),
                            "ELEVATION": float(line[31:37].strip()),
                            "STATE": line[38:40].strip(),
                            "NAME": line[38:68].strip(),
                        })
                    except (ValueError, IndexError) as e:
                        self.logger.warning(f"Failed to parse station line: {line[:30]}... Error: {e}")
                        continue
            
            if not data:
                self.logger.warning(f"No valid station data parsed from {url}")
                return pd.DataFrame()
            
            df = pd.DataFrame(data)
            self.logger.info(f"Parsed {len(df)} station records")
            return df
        except Exception as e:
            self.logger.error(f"Error parsing station data from {url}: {e}")
            raise

    def parse_inventory(self, url):

        self.logger.info(f"Fetching and parsing inventory data from {url}")
        try:
            response = requests.get(url)
            if response.status_code != 200:
                self.logger.error(f"Failed to retrieve data from {url}. Status code: {response.status_code}")
                raise ValueError(f"Failed to retrieve data from {url}. Status code: {response.status_code}")
            
            lines = response.text.split("\n")
            data = []
            for line in lines:
                if line.strip():
                    try:
                        data.append({
                            "ID": line[0:11].strip(),
                            "LATITUDE": float(line[12:20].strip()),
                            "LONGITUDE": float(line[21:30].strip()),
                            "ELEMENT": line[31:35].strip(),
                            "FIRSTYEAR": int(line[36:40].strip()),
                            "LASTYEAR": int(line[41:45].strip())
                        })
                    except (ValueError, IndexError) as e:
                        self.logger.warning(f"Failed to parse inventory line: {line[:30]}... Error: {e}")
                        continue
            
            if not data:
                self.logger.warning(f"No valid inventory data parsed from {url}")
                return pd.DataFrame()
            
            df = pd.DataFrame(data)
            self.logger.info(f"Parsed {len(df)} inventory records")
            return df
        except Exception as e:
            self.logger.error(f"Error parsing inventory data from {url}: {e}")
            raise

    def parse_countries(self, url):
 
        self.logger.info(f"Fetching and parsing country data from {url}")
        try:
            response = requests.get(url)
            if response.status_code != 200:
                self.logger.error(f"Failed to retrieve data from {url}. Status code: {response.status_code}")
                raise ValueError(f"Failed to retrieve data from {url}. Status code: {response.status_code}")
            
            lines = response.text.split("\n")
            data = []
            for line in lines:
                if line.strip():
                    try:
                        data.append({
                            "CODE": line[0:2].strip(),
                            "NAME": line[3:64].strip()
                        })
                    except (ValueError, IndexError) as e:
                        self.logger.warning(f"Failed to parse country line: {line[:30]}... Error: {e}")
                        continue
            
            if not data:
                self.logger.warning(f"No valid country data parsed from {url}")
                return pd.DataFrame()
            
            df = pd.DataFrame(data)
            self.logger.info(f"Parsed {len(df)} country records")
            return df
        except Exception as e:
            self.logger.error(f"Error parsing country data from {url}: {e}")
            raise

    def parse_states(self, url):

        self.logger.info(f"Fetching and parsing state data from {url}")
        try:
            response = requests.get(url)
            if response.status_code != 200:
                self.logger.error(f"Failed to retrieve data from {url}. Status code: {response.status_code}")
                raise ValueError(f"Failed to retrieve data from {url}. Status code: {response.status_code}")
            
            lines = response.text.split("\n")
            data = []
            for line in lines:
                if line.strip():
                    try:
                        data.append({
                            "CODE": line[0:2].strip(),
                            "NAME": line[3:50].strip()
                        })
                    except (ValueError, IndexError) as e:
                        self.logger.warning(f"Failed to parse state line: {line[:30]}... Error: {e}")
                        continue
            
            if not data:
                self.logger.warning(f"No valid state data parsed from {url}")
                return pd.DataFrame()
            
            df = pd.DataFrame(data)
            self.logger.info(f"Parsed {len(df)} state records")
            return df
        except Exception as e:
            self.logger.error(f"Error parsing state data from {url}: {e}")
            raise

    def parse_data_dly(self, line):
       
        self.logger.debug(f"Parsing daily weather data line: {line[:30]}...")
        try:
            if not line.strip():
                self.logger.warning("Empty line received for parsing")
                return {}
            
            data = []
            for i in range(21, 269, 8):
                try:
                    value = int(line[i:i+5])
                    mflag = line[i+5]
                    qflag = line[i+6]
                    sflag = line[i+7]
                    data.extend([value, mflag, qflag, sflag])
                except (ValueError, IndexError) as e:
                    self.logger.warning(f"Failed to parse data segment at position {i}: {e}")
                    data.extend([None, None, None, None])
            
            parsed_data = {
                "ID": line[0:11].strip(),
                "YEAR": int(line[11:15]),
                "Month": int(line[15:17]),
                "ELEMENT": line[17:21].strip(),
                "DATA": data
            }
            self.logger.debug(f"Parsed daily data: ID={parsed_data['ID']}, YEAR={parsed_data['YEAR']}, Month={parsed_data['Month']}, ELEMENT={parsed_data['ELEMENT']}")
            return parsed_data
        except (ValueError, IndexError) as e:
            self.logger.error(f"Error parsing daily weather data line: {e}")
            return {}