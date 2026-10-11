import csv

filename = "../Official_Project/Data/Shots_Data.csv"
rows = []
row_check = []
rows_to_clean = ['shot_aerial_won', 'shot_first_time', 'under_pressure']

with open(filename, mode="r", newline="", encoding="utf-8") as file:
    reader = csv.reader(file)
    header = next(reader)  # Save the header row

    for column in rows_to_clean:
        row_check.append(header.index(column))
    
    for row in reader:
        for i in row_check:
            if row[i] == '':
                row[i] = False
        
        rows.append(row)

with open(filename, mode="w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(header)   # Write back the header
    writer.writerows(rows)    # Write back the edited data
