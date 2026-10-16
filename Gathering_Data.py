from pathlib import Path

import pandas as pd
from statsbombpy import sb

SHOT_COLUMNS = ['shot_aerial_won', 'shot_body_part',
                'shot_end_location', 'shot_first_time', 'shot_freeze_frame',
                'shot_outcome', 'shot_statsbomb_xg', 'shot_technique',
                'shot_type', 'under_pressure', 'location']

competitions = sb.competitions()
competitions_df = competitions[competitions.competition_gender == "male"].loc[:, ['competition_id', 'season_id']]

shot_frames = []
for row in competitions_df.itertuples(index=False):
    matches_df = sb.matches(competition_id = row.competition_id, season_id = row.season_id)
    for match in matches_df.itertuples(index=False):
        events = sb.events(match_id = match.match_id)
        shots = events[events["type"] == "Shot"]
        shot_frames.append(shots.reindex(columns = SHOT_COLUMNS))

shots_df = pd.concat(shot_frames, ignore_index = True)

out_path = Path(__file__).resolve().parent.parent/ "Official_Project" / "Data" / "Shots_Data.csv"
out_path.parent.mkdir(parents = True, exist_ok = True)
shots_df.to_csv(out_path, index = False)