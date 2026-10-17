import ast
import csv
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent
shots_filename = DATA_DIR / "Shots_Data.csv"
freeze_frame_filename = DATA_DIR / "Freeze_Frame_Data.csv"
shot_rows = []
freeze_frame_rows = []

with open(shots_filename, mode = "r", newline = "", encoding = "utf-8") as file:
    reader = csv.reader(file)
    header = next(reader)

    has_shot_id = "shot_id" in header
    if not has_shot_id:
        header.insert(0, "shot_id")

    has_location_xy = "location_x" in header
    if not has_location_xy:
        location_index = header.index("location")
        header[location_index:location_index + 1] = ["location_x", "location_y"]

    has_freeze_frame = "shot_freeze_frame" in header
    if has_freeze_frame:
        freeze_frame_index = header.index("shot_freeze_frame")
        header.pop(freeze_frame_index)

    id_index = header.index("shot_id")

    for row in reader:
        if not has_shot_id:
            row.insert(0, len(shot_rows))
        shot_id = row[id_index]

        if not has_location_xy:
            row[location_index:location_index + 1] = ast.literal_eval(row[location_index])

        if has_freeze_frame:
            freeze_frame = row.pop(freeze_frame_index)
            if freeze_frame != "Penalty":
                for player in ast.literal_eval(freeze_frame):
                    x, y = player["location"]
                    goalkeeper = player["position"]["name"] == "Goalkeeper"
                    freeze_frame_rows.append([shot_id, player["teammate"], goalkeeper, x, y])

        shot_rows.append(row)

with open(shots_filename, mode = "w", newline = "", encoding = "utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(header)
    writer.writerows(shot_rows)

if has_freeze_frame:
    with open(freeze_frame_filename, mode = "w", newline = "", encoding = "utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["shot_id", "teammate", "goalkeeper", "location_x", "location_y"])
        writer.writerows(freeze_frame_rows)