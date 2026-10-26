from pathlib import Path

import duckdb as db

DATA_DIR = Path(__file__).resolve().parent / "Data"
SHOTS_PATH = DATA_DIR / "Shots_Data.csv"

MIN_DISTANCE = 1.0
NEW_COLUMNS = ["gk_distance_from_line", "gk_angle_from_center", "pressure_score"]

shots = db.read_csv(str(SHOTS_PATH), all_varchar=True)
shots = shots.project(", ".join(f'"{c}"' for c in shots.columns if c not in NEW_COLUMNS))
freeze_frames = db.read_csv(str(DATA_DIR / "Freeze_Frame_Data.csv"))

shots_with_defenders = db.sql(f"""
    WITH opposition AS (
        SELECT
            f.shot_id,
            f.goalkeeper,
            f.location_x AS player_x,
            f.location_y AS player_y,
            CAST(s.location_x AS DOUBLE) AS ball_x,
            CAST(s.location_y AS DOUBLE) AS ball_y
        FROM freeze_frames f
        JOIN shots s ON f.shot_id = CAST(s.shot_id AS BIGINT)
        WHERE NOT f.teammate
    ),

    goalkeepers AS (
        SELECT
            shot_id,
            120 - player_x AS gk_distance_from_line,
            atan2(player_y - 40, 120 - player_x) AS gk_angle_from_center
        FROM opposition
        WHERE goalkeeper
    ),

    cone_sides AS (
        SELECT
            shot_id,
            sqrt(power(player_x - ball_x, 2) + power(player_y - ball_y, 2)) AS distance,
            (120 - ball_x) * (player_y - ball_y) - (36 - ball_y) * (player_x - ball_x) AS side_1,
            8 * (120 - player_x) AS side_2,
            (ball_x - 120) * (player_y - 44) - (ball_y - 44) * (player_x - 120) AS side_3
        FROM opposition
        WHERE NOT goalkeeper
    ),

    pressure AS (
        SELECT
            shot_id,
            SUM(1 / GREATEST(distance, {MIN_DISTANCE})) AS pressure_score
        FROM cone_sides
        WHERE NOT (
            (side_1 < 0 OR side_2 < 0 OR side_3 < 0)
            AND (side_1 > 0 OR side_2 > 0 OR side_3 > 0)
        )
        GROUP BY shot_id
    )

    SELECT
        s.*,
        ROUND(g.gk_distance_from_line, 4) AS gk_distance_from_line,
        ROUND(g.gk_angle_from_center, 4) AS gk_angle_from_center,
        ROUND(COALESCE(p.pressure_score, 0), 4) AS pressure_score
    FROM shots s
    LEFT JOIN goalkeepers g ON g.shot_id = CAST(s.shot_id AS BIGINT)
    LEFT JOIN pressure p ON p.shot_id = CAST(s.shot_id AS BIGINT)
    ORDER BY CAST(s.shot_id AS BIGINT)
""").df()

shots_with_defenders.to_csv(SHOTS_PATH, index=False)
