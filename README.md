# Tile Classification Service — Thin Slice (Part 2)

Classifies satellite tiles by land-use type (7 classes: AnnualCrop, Forest,
Highway, Industrial, Residential, River, SeaLake) and stores each result.
Runs fully offline at inference time.

## Approach

- **Model:** a small CNN (4 conv blocks: 32→64→128→256 filters, BatchNorm +
  ReLU + MaxPool each, global average pooling, dropout, ~423K parameters),
  trained from scratch (no pretrained weights) on the 1050 labeled
  `candidate_tiles`, using GPU in Google Colab (`galaxeye_train_cnn.ipynb`).
- Deliberately shallow/small given the limited dataset (~150 images/class) —
  a deeper network would overfit. Data augmentation (horizontal + vertical
  flips, rotation, mild color jitter) compensates for limited data; flips
  in both axes are safe since satellite tiles have no fixed "up" direction.
- Achieved 87.9% validation accuracy and **89.5% accuracy on `eval_set`**
  (ground truth, never touched during training).
- Weakest classes: Highway and Industrial get confused with each other
  more than with any other class — both are long, thin, linear features
  that look visually similar at 64×64 resolution.

## Setup

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

`artifacts_cnn/tile_cnn.pt` and `artifacts_cnn/classes.json` (the trained
model) are already included — no training step is required to run the
service. To retrain, use `galaxeye_train_cnn.ipynb` in Google Colab.

## Run

```bash
uvicorn app:app --reload
```

Then classify a tile:
```bash
curl -X POST http://127.0.0.1:8000/classify -F "file=@C:\path\to\some_tile.png"
```

Results are stored in `results.db` (SQLite), table `classifications`.

## Offline behavior

The model was trained from scratch (no pretrained/downloaded weights), so
there are zero network calls at any point — `tile_cnn.pt` loads directly
from local disk on startup.

## What's implemented vs. stubbed

**Implemented (the required core path):**
- `POST /classify` — full path: accept image → classify → store → return result
- SQLite storage of every result, including full per-class probabilities
  and a model version tag

**Deliberately stubbed / not built:**
- No query/listing endpoints (`GET /results`, filtering by class or
  confidence, etc.) — described in the design note, not implemented
- No auth, no rate limiting, no batch upload
- No automatic retraining or model-drift detection
- No containerization/deployment config

## Files

- `cnn_model.py` — CNN architecture + loading/inference wrapper
- `artifacts_cnn/tile_cnn.pt` — trained weights
- `artifacts_cnn/classes.json` — class name order
- `galaxeye_train_cnn.ipynb` — Colab notebook used to train the model
- `db.py` — SQLite schema + insert helper
- `app.py` — the FastAPI service
- `results.db` — created on first API request