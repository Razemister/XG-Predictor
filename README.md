# xG Shot Model

An expected goals (xG) model that estimates the probability a shot becomes a goal using only **where the ball and players are** at the moment of the shot. Built on StatsBomb open data with SQL queries in DuckDB, scikit-learn models, and (coming soon) an interactive Streamlit site.

> **Status:** In progress. Data pipeline, baseline model and both freeze-frame models are complete; calibration checks and the website are under development.
>
> **Live demo:** _coming soon_

---

## The question

How much does knowing where the defenders and goalkeeper are improve a shot's goal probability, compared with knowing only where the shot was taken from?

**Short answer so far:** a lot. Adding freeze-frame features cuts log loss from 0.2909 to 0.2634 and lifts ROC AUC from 0.743 to 0.807 (see [Results](#results)).

## Scope

The model uses **positional information plus how the shot was struck**:

| Input | Description |
|---|---|
| Ball position | Shot location on the pitch, plus distance and angle to goal |
| Player positions | Defenders and goalkeeper at the moment of the shot (StatsBomb freeze frames), turned into the features below |
| Body part | Right foot, left foot, head or other |
| Shot details | First-time shot, technique (normal, volley, lob, …) and shot type (open play, free kick, corner) |

No match context, competition, team or player identity is used as a model input.

## Pipeline

```
StatsBomb open data  →  CSV files  →  DuckDB (validation, features)  →  scikit-learn  →  Streamlit
```

1. **Load:** `Data/Gathering_Data.py` pulls shot events from all men's competitions in StatsBomb open data and saves them to `Data/Shots_Data.csv`.
2. **Clean:** `Data/Clearing_Blanks.py` fills blank values, and `Data/Player_Info_CSV_Maker.py` adds a `shot_id`, splits the shot location into `location_x` / `location_y`, and moves each shot's freeze frame into `Data/Freeze_Frame_Data.csv` (one row per player).
3. **Validate:** `Data_Validation.py` uses DuckDB SQL to check both CSVs for missing player or shot locations.
4. **Freeze-frame features:** `Defenders_Data_Preprocessing.py` joins every shot to its freeze frame in DuckDB SQL and adds the goalkeeper and pressure features (described below) as new columns in `Data/Shots_Data.csv`.
5. **Distance and angle:** A DuckDB SQL query in `Comparision.py` computes distance and angle to goal for every shot.
6. **Model:** `Comparision.py` calls the three models (`Logistic_Regression.py` for the baseline, `Freeze_Frame.py` for the freeze-frame models), prints each model's log loss, ROC AUC and best settings, and writes out-of-fold predictions to `Data/`.
7. **Evaluate (planned):** Calibration buckets and comparison against StatsBomb's own xG.
8. **Serve (planned):** A Streamlit site lets users place the ball, defenders and goalkeeper on a pitch and get an xG estimate.

## Pitch geometry

All features use StatsBomb's coordinate system: the pitch is 120 × 80 yards, the attacking goal sits on the line `x = 120`, and the posts are at `y = 36` and `y = 44` (an 8-yard goal centred on `y = 40`).

- **Distance to goal:** straight-line distance from the ball to the centre of goal, `(120, 40)`.
- **Angle to goal:** the angle the goal mouth makes from the ball, i.e. how much of the goal the shooter can see:

$$\text{angle} = \arctan\!\left(\frac{8\,(120 - x)}{(120 - x)^2 + (y - 40)^2 - 16}\right)$$

## Freeze-frame features

A StatsBomb freeze frame records where every visible player was at the moment of the shot (about 13 players per shot on average). Each player row says whether they are a teammate and whether they are the goalkeeper. Teammates are ignored; only **opponents** are used. Every one of the 65,822 shots has a freeze frame.

### 1. Goalkeeper distance from the goal line

```
gk_distance_from_line = 120 − gk_x
```

How far the goalkeeper has come off the line, in yards. A keeper who has come out narrows the angle but leaves the goal exposed to lobs. The median is 2.0 yards.

### 2. Goalkeeper offset from the centre of goal

```
gk_angle_from_center = atan2(gk_y − 40, 120 − gk_x)
```

The goalkeeper's angle from the centre of the goal, seen from the goal line, in radians. `0` means the keeper is dead centre; values towards ±π/2 mean they are far over to one side. Using an angle rather than a raw y-offset means a keeper who is off-centre but well off their line is treated differently from one who is off-centre on the line.

58 shots have no goalkeeper in the freeze frame (e.g. the keeper was out of the frame). Their two goalkeeper features are left empty in the data and filled with the median inside the model pipeline.

### 3. The shooting triangle

The shooting triangle (or cone) is the triangle with corners at the **ball** and the **two posts**. Any defender inside it is physically between the ball and the goal and can block the shot.

To test whether a defender at `(px, py)` is inside the triangle, the SQL uses the sign of the 2D cross product along each of the triangle's three edges:

| Edge | Cross product |
|---|---|
| Ball → near post `(120, 36)` | `(120 − bx)(py − by) − (36 − by)(px − bx)` |
| Post to post (goal line) | `8 (120 − px)` |
| Far post `(120, 44)` → ball | `(bx − 120)(py − 44) − (by − 44)(px − 120)` |

A point is inside the triangle when all three values have the **same sign** (on the boundary counts as inside). If some are positive and some negative, the defender is outside. The goalkeeper is handled separately by the features above, so only outfield defenders are tested.

About half of all shots (51%) have at least one defender inside the triangle.

### 4. Pressure score

The pressure score combines **how many** defenders are in the triangle with **how close** they are to the ball:

$$\text{pressure\_score} = \sum_{\text{defenders in triangle}} \frac{1}{\max(d_i,\ 1)}$$

where $d_i$ is the distance in yards from defender $i$ to the ball.

- **Each defender adds to the score**, so two blockers produce more pressure than one.
- **Closer defenders count more:** a defender 2 yards away adds 0.5, one 10 yards away adds only 0.1. A defender right in front of the ball is far more likely to block than one standing near the goal line.
- **The 1-yard floor** (`MIN_DISTANCE`) stops a defender almost on top of the ball from producing a near-infinite value.
- **No defenders in the triangle** gives a score of 0 (an open shot).

Across all shots the median pressure score is 0.06 and the maximum is 2.8.

### Features considered but not used

- Distance to the nearest defender, anywhere on the pitch
- Number of opponents within 3–5 yards of the ball

The pressure score covers most of what these would add for blocking: it already counts defenders and weights them by distance, but only the ones actually in the shot's path.

## Models

| Model | Features | Status |
|---|---|---|
| Baseline: logistic regression | Distance and angle to goal (ball position only) | ✅ Done |
| Freeze-frame: logistic regression | Baseline + ball x/y + freeze-frame features + body part + shot details | ✅ Done |
| Freeze-frame: gradient boosting | Same as above | ✅ Done |

### Freeze-frame pipeline

Both freeze-frame models share one scikit-learn `Pipeline` (in `Freeze_Frame.py`), so every step is fitted only on the training folds:

| Step | Columns |
|---|---|
| **Dropped** | `shot_id`, `shot_outcome` (it is the target), `shot_statsbomb_xg` (StatsBomb's own model, so it would leak the answer), `shot_end_location` (only known after the shot), `shot_aerial_won`, `under_pressure` |
| **One-hot encoded** | `shot_body_part` |
| **Ordinal encoded** | `shot_first_time`, `shot_technique`, `shot_type` |
| **Numeric** (median-imputed; also standardised for logistic regression) | `location_x`, `location_y`, `distance`, `angle`, `gk_distance_from_line`, `gk_angle_from_center`, `pressure_score` |

### Tuning

Each model is tuned with `GridSearchCV` on the 5 cross-validation folds, picking the settings with the lowest log loss. I compared training loss with validation loss for every setting to find the point between underfitting and overfitting.

**Logistic regression:** searched regularisation strength `C` (0.001 to 100) and L1 vs L2 penalty.
- Best: **C = 1, L2 penalty.**
- Training and validation loss differ by only about 0.0002, so the model is not overfitting; it is limited by being linear. Strong regularisation (`C = 0.001`) underfits (log loss 0.273–0.284).

**Gradient boosting** (`HistGradientBoostingClassifier`): searched learning rate, number of trees, tree depth and minimum samples per leaf (48 combinations). Early stopping is turned off so the number of trees is really tuned (scikit-learn turns it on automatically for datasets over 10,000 rows).
- Best: **500 trees, learning rate 0.03, max depth 3, min 20 samples per leaf.**
- Large models overfit badly: with learning rate 0.1, depth 6 and 500 trees, training loss drops to 0.18 but validation loss rises to 0.276.
- Small models underfit: 100 trees at learning rate 0.03 only reach 0.268.
- Because 500 was the largest tree count in the grid, I also tried 700, 1000 and 1500 trees. Validation loss got steadily worse (0.2635 → 0.2649), confirming 500 is the best point.

### Validation

All models are scored with **5-fold stratified cross-validation** (`StratifiedKFold`, shuffled, `random_state=42`), so every fold has the same share of goals (about 10%). Reported scores are the average over the 5 folds. Predictions saved to `Data/` are out-of-fold: each shot's xG comes from a model that never saw that shot.

The best settings are chosen on the same folds that produce the scores, so the scores may be very slightly optimistic. The differences between the top settings are small, so the effect should be tiny.

**Planned:** split **by match**, so no game contributes shots to both training and validation. Shots from the same game share conditions (goalkeeper, weather, game state), and a random split lets the model see near-copies of its test shots during training. This needs `match_id` to be saved during data gathering.

## Results

| Model | Log loss ↓ | ROC AUC ↑ |
|---|---|---|
| Baseline (ball position) | 0.2909 | 0.7428 |
| Freeze-frame (logistic regression) | 0.2659 | 0.8031 |
| Freeze-frame (gradient boosting) | **0.2634** | **0.8074** |

- **The freeze frame is where most of the gain comes from.** Going from the baseline to freeze-frame logistic regression improves log loss by 0.025 and ROC AUC by 0.06.
- **Gradient boosting adds a smaller improvement** on top (log loss −0.0025, ROC AUC +0.004). It can pick up interactions a linear model can't, such as pressure mattering more for close-range shots.

Dataset: **65,822 shots** (6,667 goals, 10.1%) from **70+ men's competitions**. Penalties are not in the data; only open play, free kicks and corners.

## Repository structure

```
├── Data/
│   ├── Gathering_Data.py                  # Downloads men's shots from StatsBomb
│   ├── Clearing_Blanks.py                 # Fills blank values
│   ├── Player_Info_CSV_Maker.py           # Adds shot IDs, splits locations, builds freeze-frame CSV
│   ├── Shots_Data.csv                     # One row per shot (not committed)
│   ├── Freeze_Frame_Data.csv              # One row per player per shot (not committed)
│   ├── Logistic_Prediction.csv            # Baseline out-of-fold predictions (not committed)
│   ├── Freeze_Frame_Log_Prediction.csv    # Freeze-frame logistic predictions (not committed)
│   └── Freeze_Frame_GB_Prediction.csv     # Freeze-frame gradient boosting predictions (not committed)
├── Data_Validation.py                     # Missing-location checks in DuckDB SQL
├── Defenders_Data_Preprocessing.py        # Goalkeeper, shooting-triangle and pressure features in DuckDB SQL
├── Comparision.py                         # Builds distance/angle, runs all three models, prints results
├── Logistic_Regression.py                 # Baseline logistic regression
├── Freeze_Frame.py                        # Freeze-frame pipeline, logistic regression and gradient boosting
└── README.md
```

## Running it locally

```bash
git clone https://github.com/Razemister/XG-Predictor.git
cd XG-Predictor
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install pandas duckdb scikit-learn statsbombpy
```

Then run the scripts in order:

```bash
python Data/Gathering_Data.py
python Data/Clearing_Blanks.py
python Data/Player_Info_CSV_Maker.py
python Data_Validation.py
python Defenders_Data_Preprocessing.py
python Comparision.py              # takes a few minutes: both freeze-frame models run a grid search
```

## Roadmap

- [x] Load men's shots from StatsBomb into CSV files
- [x] Data quality checks with DuckDB SQL
- [x] Distance and angle features in SQL
- [x] Baseline logistic regression
- [x] Freeze-frame features (goalkeeper position, shooting triangle, pressure score)
- [x] Freeze-frame logistic regression and gradient boosting, tuned with cross-validation
- [ ] Calibration and comparison with StatsBomb xG
- [ ] Streamlit site: xG calculator, similar shots, insights
- [ ] Deploy

## Limitations

- StatsBomb open data is a set of selected releases, not complete league coverage, so some competitions and eras are over-represented.
- Freeze frames only include players visible on the broadcast footage, so a defender off-camera is missing from the triangle and pressure score.
- The pressure score only counts defenders inside the triangle. A defender just outside it, about to close the shooter down, adds nothing.
- Shots from the same match can appear in both training and validation folds until grouped cross-validation is added.

## Data

Data provided by [StatsBomb](https://github.com/statsbomb/open-data) under their open data licence.

## Tech

Python · pandas · DuckDB (SQL) · scikit-learn · statsbombpy · Streamlit (planned)
