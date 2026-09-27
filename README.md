# Tile Classification Service --- Thin Slice (Part 2)

A small offline service for classifying satellite image tiles into 7
land-use classes:

`AnnualCrop, Forest, Highway, Industrial, Residential, River, SeaLake`

The service accepts a tile image, runs the local CNN model, stores the
prediction in SQLite, and returns the result.

## Approach

### Model

I used a small CNN trained from scratch rather than a pretrained
ImageNet model.

The model has 4 convolution blocks with:

`32 → 64 → 128 → 256` filters

Each block uses BatchNorm, ReLU and MaxPool, followed by global average
pooling and a small classifier head. The model has approximately 423K
parameters.

The model was trained on the 1050 labelled `candidate_tiles` using GPU
in Google Colab.

Since the dataset is relatively small, I kept the model small and used
data augmentation during training:

-   horizontal flip
-   vertical flip
-   small rotation
-   mild brightness/contrast changes

The flips are reasonable for this dataset because satellite tiles do not
have a fixed "up" direction.

### Evaluation

The model achieved:

-   Validation accuracy: **87.9%**
-   Evaluation-set accuracy: **89.5%**

The evaluation set was kept separate from training.

Highway and River were the classes with more confusion between them.
At 64×64 resolution, both can contain long and linear visual
structures, which can make them harder to distinguish.

## Setup

Create and activate a virtual environment:

``` bash
python -m venv venv
```

On Windows Command Prompt:

``` bash
venv\Scripts\activate
```

Install the dependencies:

``` bash
pip install -r requirements.txt
```

The trained model files are already included:

``` text
artifacts_cnn/
├── tile_cnn.pt
└── classes.json
```

Therefore, no training or internet connection is required to run the
service.

## Run

Start the FastAPI application:

``` bash
uvicorn app:app --reload
```

The API will be available at:

``` text
http://127.0.0.1:8000
```

Swagger documentation is available at:

``` text
http://127.0.0.1:8000/docs
```

## Classify a tile

Using curl:

``` bash
curl -X POST http://127.0.0.1:8000/classify -F "file=@C:\path\to\some_tile.png"
```

The response contains:

-   prediction ID
-   filename
-   predicted class
-   confidence
-   probabilities for all seven classes
-   model version

Example:

``` json
{
  "id": 2,
  "filename": "tile_031.png",
  "predicted_class": "River",
  "confidence": 0.9864,
  "all_class_probabilities": {
    "AnnualCrop": 0.000008,
    "Forest": 0.000001,
    "Highway": 0.0135,
    "Industrial": 0.000000,
    "Residential": 0.000001,
    "River": 0.9864,
    "SeaLake": 0.000003
  },
  "model_version": "scratch-cnn-v1"
}
```

## Offline behavior

The trained model and class mapping are stored locally in the
`artifacts_cnn` directory.

At inference time, the service loads the model directly from local disk
and does not call any external API or download model files.

This allows the classification service to run on isolated hardware
without internet access.

## What's implemented

The current thin slice implements the main classification path:

``` text
Upload tile
    ↓
Preprocess image
    ↓
Run local CNN
    ↓
Generate prediction + probabilities
    ↓
Store result in SQLite
    ↓
Return JSON response
```

Implemented:

-   `POST /classify`
-   Image validation and preprocessing
-   Local CNN inference
-   Full probability output
-   SQLite storage
-   Model version stored with each result

## Not implemented

The following are intentionally outside the current thin slice:

-   Query/listing endpoints
-   Filtering historical results
-   Authentication
-   Rate limiting
-   Batch upload
-   Automatic retraining
-   Model-drift detection
-   Containerization/deployment configuration

These are discussed as future/full-system considerations in the design
note.

## Storage

Results are stored in:

``` text
results.db
```

The database contains a `classifications` table with the prediction,
confidence, complete probability vector, model version and timestamp.

The database file is created automatically when the service first stores
a classification result.

## Files

`app.py`

FastAPI application and `/classify` endpoint.

`cnn_model.py`

CNN architecture, model loading and inference logic.

`db.py`

SQLite schema and result-storage functions.

`artifacts_cnn/tile_cnn.pt`

Trained CNN weights.

`artifacts_cnn/classes.json`

Class names used by the model.

`requirements.txt`

Python dependencies required to run the service.

`GalaxEye Design Note — [Sangeeta_Upadhye].pdf`

Part 1 design note describing the architecture, design decisions,
assumptions and trade-offs.
