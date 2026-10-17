# xG Shot Model

An expected goals (xG) model that estimates the probability a shot becomes a goal using only **where the ball and players are** at the moment of the shot. Built on StatsBomb open data with SQL queries in DuckDB, scikit-learn models, and (coming soon) an interactive Streamlit site.

> **Status:** In progress. Data pipeline and baseline model are complete; freeze-frame model and website are under development.
>
> **Live demo:** _coming soon_

---

## The question

How much does knowing where the defenders and goalkeeper are improve a shot's goal probability, compared with knowing only where the shot was taken from?

## Scope

The model uses **positional information only**:

| Input | Description |
|---|---|
| Ball position | Shot location on the pitch |
| Player positions | Teammates, defenders and goalkeeper at the moment of the shot (StatsBomb freeze frames) |
| Body part | Foot, head or other |

No match context, competition, team or player identity is used as a model input.

## Pipeline

```
StatsBomb open data  →  CSV files  →  DuckDB (validation, features)  →  scikit-learn  →  Streamlit
```

1. **Load:** `Data/Gathering_Data.py` pulls shot events from all men's competitions in StatsBomb open data and saves them to `Data/Shots_Data.csv`.
2. **Clean:** `Data/Clearing_Blanks.py` fills blank values, and `Data/Player_Info_CSV_Maker.py` adds a `shot_id`, splits the shot location into `location_x` / `location_y`, and moves each shot's freeze frame into `Data/Freeze_Frame_Data.csv` (one row per player).
3. **Validate:** `Data_Validation.py` uses DuckDB SQL to check both CSVs for missing player or shot locations.
4. **Engineer features:** A DuckDB SQL query in `Comparision.py` computes distance and angle to goal for every shot.
5. **Model:** `Logistic_Regression.py` trains the model with scikit-learn, writes test-set predictions to `Data/Linear_Prediction.csv`, and returns log loss and ROC AUC.
6. **Evaluate:** Compare models (calibration buckets, log loss) and against StatsBomb's own xG.
7. **Serve:** A Streamlit site lets users place the ball, defenders and goalkeeper on a pitch and get an xG estimate.

## Models

| Model | Features | Status |
|---|---|---|
| Baseline: logistic regression | Distance and angle to goal (ball position only) | ✅ Done |
| Freeze-frame: logistic regression | Baseline + freeze-frame features + body part | 🔄 In progress |
| Freeze-frame: gradient boosting | Same as above | ⏳ Planned |

**Planned freeze-frame features**
- Defenders inside the shooting cone (the triangle from the ball to both posts)
- Goalkeeper distance from the goal line and offset from the centre of goal
- Distance to the nearest defender
- Number of opponents within 3–5 m of the ball

### Train/test split

Shots are currently split randomly into 80% train and 20% test, stratified so both sets have the same share of goals.

**Planned:** split **by match**, so no game contributes shots to both sets. Shots from the same game share conditions (goalkeeper, weather, game state), and a random split lets the model see near-copies of its test shots during training. This needs `match_id` to be saved during data gathering.

## Results

| Model | Log loss ↓ | ROC AUC ↑ |
|---|---|---|
| Baseline (ball position) | 0.309 | 0.746 |
| Freeze-frame (logistic) | _TBD_ | _TBD_ |
| Freeze-frame (gradient boosting) | _TBD_ | _TBD_ |

Dataset: **66856 shots** from **70+ men's competitions**.

## Repository structure

```
├── Data/
│   ├── Gathering_Data.py          # Downloads men's shots from StatsBomb
│   ├── Clearing_Blanks.py         # Fills blank values
│   ├── Player_Info_CSV_Maker.py   # Adds shot IDs, splits locations, builds freeze-frame CSV
│   ├── Shots_Data.csv             # One row per shot (not committed)
│   ├── Freeze_Frame_Data.csv      # One row per player per shot (not committed)
│   └── Linear_Prediction.csv      # Baseline test-set predictions (not committed)
├── Data_Validation.py             # Missing-location checks in DuckDB SQL
├── Comparision.py                 # Builds features, splits data, runs the models
├── Logistic_Regression.py         # Baseline logistic regression
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
python Comparision.py
```

## Roadmap

- [x] Load men's shots from StatsBomb into CSV files
- [x] Data quality checks with DuckDB SQL
- [x] Distance and angle features in SQL
- [x] Baseline logistic regression
- [ ] Grouped (by-match) train/test split
- [ ] Freeze-frame features and model
- [ ] Gradient boosting comparison and body-part ablation
- [ ] Calibration and comparison with StatsBomb xG
- [ ] Streamlit site: xG calculator, similar shots, insights
- [ ] Deploy

## Limitations

- StatsBomb open data is a set of selected releases, not complete league coverage, so some competitions and eras are over-represented.
- Some shots, especially in older data, have no freeze frame and are excluded from the freeze-frame model.
- Penalties are currently included in the baseline. They should be excluded, because player positions barely affect them.

## Data

Data provided by [StatsBomb](https://github.com/statsbomb/open-data) under their open data licence.

## Tech

Python · pandas · DuckDB (SQL) · scikit-learn · statsbombpy · Streamlit (planned)
