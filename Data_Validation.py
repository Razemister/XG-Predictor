import duckdb as db

shots = db.read_csv("../Official_project/Data/Shots_Data.csv")
player = db.read_csv("../Official_project/Data/Freeze_Frame_Data.csv")

null_locations_shots = db.sql("SELECT COUNT(*) FROM shots WHERE location_x IS NULL OR location_y IS NULL")
null_locations_player = db.sql("SELECT COUNT(*) FROM player WHERE location_x IS NULL OR location_y IS NULL")

valid_data = (null_locations_shots == 0) and (null_locations_player == 0)