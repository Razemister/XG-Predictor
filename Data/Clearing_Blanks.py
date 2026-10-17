import csv
from pathlib import Path

filename = Path(__file__).resolve().parent / "Shots_Data.csv"
rows = []
row_check = []
rows_to_clean = ['shot_aerial_won', 'shot_first_time', 'under_pressure', 'shot_freeze_frame']

with open(filename, mode="r", newline="", encoding="utf-8") as file:
    reader = csv.reader(file)
    header = next(reader)

    for column in rows_to_clean:
        if column in header:
            row_check.append(header.index(column))

    for row in reader:
        for i in row_check:
            if row[i] == '':
                if header[i] != 'shot_freeze_frame':
                    row[i] = False
                else:
                    row[i] = "Penalty"
        
        rows.append(row)

with open(filename, mode="w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(header)
    writer.writerows(rows)
