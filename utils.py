import csv
import os
from datetime import datetime

FILENAME = 'database/output.csv'


def write_to_csv(count, include_header=False):
    file_exists = os.path.exists(FILENAME)
    with open(FILENAME, mode='a+', newline='') as csvfile:
        writer = csv.writer(csvfile)

        if not file_exists or include_header:
            writer.writerow(['people_count','date', 'time'])

        current_datetime = datetime.now()
        date_str = current_datetime.strftime("%Y-%m-%d")
        time_str = current_datetime.strftime("%H:%M:%S")
        writer.writerow([count, date_str, time_str])
