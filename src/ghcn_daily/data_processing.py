class DataProcessor:
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
