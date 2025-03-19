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

    @staticmethod
    def process_data(parsed_data):
        """
        Process the parsed data, for example, applying transformations or calculations.
        This is an example, adjust it based on your needs.

        :param parsed_data: A dictionary with parsed data (from `parse_data_dly`)
        :return: Processed data, could be the same or modified
        """
        # For example, you might want to apply a transformation to the data
        # For now, we'll just return the parsed_data as it is.
        return parsed_data
