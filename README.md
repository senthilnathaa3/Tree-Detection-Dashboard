# Tree Detection Dashboard

An analytics dashboard that visualizes results from a trained ResNet18 deep learning model built on the **TreeSat Benchmark Sentinel Dataset** (200m S1 + S2 only).

---

### Architecture

```
TREESATAI/
├── backend/                    # FastAPI ML inference service
│   ├── main.py                # FastAPI app entry point
│   ├── model_loader.py        # PyTorch model definition & singleton
│   ├── inference.py           # Tensor preprocessing & inference
│   ├── metrics.py             # Ecological metrics computation
│   ├── utils.py               # File handling & image conversion
│   ├── checkpoints/           # Model weights (best_model.pth)
│   └── routes/
│       └── api.py             # API endpoint definitions
├── frontend/                   # React + Vite dashboard
│   ├── src/
│   │   ├── App.jsx            # Main dashboard layout
│   │   ├── index.css          # Design system & styles
│   │   ├── components/
│   │   │   ├── Sidebar.jsx          # Upload & settings panel
│   │   │   ├── KPICards.jsx         # Animated KPI metrics
│   │   │   ├── TreeAnalytics.jsx    # Tree count charts
│   │   │   ├── SpeciesAnalytics.jsx # Species distribution
│   │   │   ├── BiodiversityMetrics.jsx # Diversity indices
│   │   │   └── SpatialHeatmaps.jsx  # Spatial heatmaps
│   │   └── services/
│   │       └── api.js         # API service layer
│   └── package.json
├── generate_test_data.py      # Test .tif generator
└── README.md
```

### Quick Start

#### 1. Generate test data
```bash
python3 generate_test_data.py
```

#### 2. Start the backend
```bash
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

#### 3. Start the frontend
```bash
cd frontend && npm install && npm run dev
```

#### 4. Open dashboard
Navigate to `http://localhost:5173`

---

### Model Details

- **Architecture**: ResNet18 Multi-Head
- **Input**: 15 channels (13 Sentinel-2 + 2 Sentinel-1)
- **Resolution**: 200m patches
- **Head 1**: Tree density regression (sigmoid, 0-1)
- **Head 2**: Multi-label species classification (20 species)
- **Checkpoint**: `backend/checkpoints/best_model.pth`
- **Model Variant Switch**:
  - `MODEL_VARIANT=v1` (default, ResNet18 baseline)
  - `MODEL_VARIANT=v2` (spectral stem + ResNet34 + stronger heads)

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/upload` | Upload .tif file |
| POST | `/api/predict` | Run inference |
| POST | `/api/preview` | Generate RGB/NDVI preview |
| GET | `/api/batch-stats` | Aggregated batch statistics |
| GET | `/api/health` | Health check |
| GET | `/api/species-labels` | Species label list |

### Offline Evaluation and Threshold Tuning

You can now evaluate model outputs against a ground-truth CSV and tune per-species thresholds.

#### Ground-truth CSV schema

- Required: `filename`
- Optional density target: `density` or `tree_density`
- Species targets (binary 0/1), either:
  - `species_<label>` columns (example: `species_Abies`), or
  - direct label columns (`Abies`, `Acer`, ...).

#### Run from CLI

```bash
python3 evaluate_model.py \
  --dataset-path /path/to/dataset \
  --ground-truth-csv /path/to/ground_truth.csv \
  --species-threshold 0.5
```

Outputs are saved under `backend/outputs/evaluation_<timestamp>/`:
- `evaluation_report.json`
- `per_tile_predictions.csv`
- `tuned_thresholds.csv`

#### Run from API

`POST /api/evaluate-offline`

```json
{
  "dataset_path": "/path/to/dataset",
  "ground_truth_csv": "/path/to/ground_truth.csv",
  "species_threshold": 0.5,
  "threshold_grid": [0.1, 0.2, 0.3, 0.4, 0.5]
}
```

### Lat/Lon-First External Validation

Use one endpoint to enter `lat/lon` and validate model output against FIA or ESA WorldCover.

`POST /api/validate-location`

#### FIA mode

```json
{
  "lat": 38.7,
  "lon": -79.9,
  "radius_km": 10,
  "provider": "planetary_computer",
  "start_date": "2024-06-01",
  "end_date": "2024-08-31",
  "cloud_cover_max": 30,
  "threshold": 0.5,
  "validation_source": "fia",
  "fia_csv_path": "/path/to/fia_export.csv",
  "year_start": 2018,
  "year_end": 2024
}
```

#### ESA WorldCover mode

```json
{
  "lat": 38.7,
  "lon": -79.9,
  "radius_km": 10,
  "provider": "planetary_computer",
  "start_date": "2024-06-01",
  "end_date": "2024-08-31",
  "threshold": 0.5,
  "validation_source": "esa_worldcover",
  "worldcover_path": "/path/to/ESA_WorldCover_*.tif"
}
```

Behavior:
- Builds AOI bbox from `lat/lon + radius_km`.
- If `dataset_path` is provided: runs model on local S1/S2 tiles intersecting AOI.
- If `dataset_path` is omitted: fetches Sentinel-2 L2A + Sentinel-1 RTC from Planetary Computer using `start_date/end_date`, runs model on fetched data.
- Computes external validation summary and model-vs-external consistency report.

### Convert FIA DataMart to Validator CSV

If you download FIA DataMart ZIP files, convert them with:

`POST /api/convert-fia-datamart`

```json
{
  "source_path": "/path/to/datamart_state_zip_or_folder",
  "output_csv_path": "/path/to/fia_converted.csv"
}
```

Notes:
- `PLOT.csv` is required.
- `TREE.csv` is optional; if present, it is used to compute `trees_per_acre` and dominant species.
- Output CSV columns: `source_plot_id, lat, lon, year, species, trees_per_acre, trees_per_hectare`.

### Fit and Apply FIA Calibration

To reduce model-vs-FIA bias, fit linear calibration from historical AOI runs.

1) Prepare a CSV with columns:
- `model_tph`
- `fia_tph`

2) Fit calibration:

`POST /api/fit-fia-calibration`

```json
{
  "calibration_csv_path": "/path/to/calibration_samples.csv"
}
```

Response gives:
- `fit.slope`
- `fit.intercept`

3) Apply calibration in location validation:

`POST /api/validate-location` with extra fields:
- `calibration_slope`
- `calibration_intercept`

The response then includes calibrated agreement fields:
- `comparison.density_agreement_calibrated` (FIA mode)
