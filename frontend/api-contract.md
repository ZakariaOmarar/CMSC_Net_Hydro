# Hydropower Anomaly Detection API Contract

Base URL: `http://localhost:8050`

All endpoints return JSON. Request bodies use `Content-Type: application/json`.

---

## Health

### `GET /api/health`

Check API availability.

**Response:**
```json
{ "status": "ok" }
```

### `GET /api/config`

Return current pipeline configuration.

**Response:** Full `PipelineConfig` object (varies by configuration).

---

## Simulation

### `POST /api/simulate`

Generate simulated sensor data and load into memory.

**Request body:**
```json
{
  "duration_s": 60.0,
  "scenario": "normal",
  "format": "udbf"
}
```

| Field        | Type   | Default    | Description                          |
|-------------|--------|------------|--------------------------------------|
| `duration_s` | float  | `60.0`     | Simulation duration in seconds       |
| `scenario`   | string | `"normal"` | Scenario name (`normal`, `anomalous`) |
| `format`     | string | `"udbf"`   | Output format (`udbf`, `csv`)        |

**Response:** Array of segment summaries.
```json
[
  {
    "id": "seg_0",
    "start_time": "2025-01-01T00:00:00",
    "duration_s": 10.0,
    "state_code": "TU",
    "n_mic_samples": 500000,
    "n_accel_samples": 100000,
    "mic_sample_rate": 50000,
    "accel_sample_rate": 10000
  }
]
```

---

## Ingestion

### `POST /api/ingest`

Load sensor data from a directory on the server filesystem.

**Request body:**
```json
{
  "data_dir": "/path/to/data"
}
```

**Response:** Array of segment summaries (same shape as `/api/simulate`).

---

## Segments

### `GET /api/segments`

List all loaded segments.

**Response:** Array of segment summaries (same shape as `/api/simulate`).

### `GET /api/segments/{seg_id}/waveform`

Get time-domain waveform data for a segment.

**Query parameters:**

| Param       | Type   | Default | Description                     |
|------------|--------|---------|----------------------------------|
| `sensor`    | string | `"mic"` | `"mic"` or `"accel"`            |
| `channel`   | int    | `0`     | Channel index (0-based)         |
| `max_points`| int    | `5000`  | Maximum number of points (100-100000) |

**Response:**
```json
{
  "times": [0.0, 0.0001, ...],
  "values": [0.123, -0.456, ...],
  "sample_rate": 50000,
  "channel": 0,
  "sensor": "mic"
}
```

### `GET /api/segments/{seg_id}/spectrum`

Get FFT spectrum for a segment.

**Query parameters:**

| Param       | Type   | Default | Description                     |
|------------|--------|---------|----------------------------------|
| `sensor`    | string | `"mic"` | `"mic"` or `"accel"`            |
| `channel`   | int    | `0`     | Channel index (0-based)         |
| `max_points`| int    | `4096`  | Max FFT size (64-65536)         |

**Response:**
```json
{
  "freqs": [0.0, 1.22, ...],
  "magnitude": [0.001, 0.023, ...]
}
```

### `GET /api/segments/{seg_id}/spectrogram`

Get spectrogram (time-frequency power) for a segment.

**Query parameters:**

| Param    | Type   | Default | Description              |
|---------|--------|---------|--------------------------|
| `sensor` | string | `"mic"` | `"mic"` or `"accel"`     |
| `channel`| int    | `0`     | Channel index (0-based)  |

**Response:**
```json
{
  "times": [0.0, 0.02, ...],
  "freqs": [0.0, 48.8, ...],
  "power": [[0.001, ...], ...]
}
```

### `GET /api/segments/{seg_id}/features`

Extract and return features for a segment.

**Response:** Object with feature names as keys and numeric values.
```json
{
  "mic_rms_0": 0.0234,
  "accel_peak_3": 1.456,
  ...
}
```

### `GET /api/segments/{seg_id}/detection`

Run detection on a single segment.

**Response:**
```json
{
  "segment_id": "seg_0",
  "state_code": "TU",
  "state_confidence": 0.95,
  "is_anomaly": false,
  "anomaly_score": 0.12,
  "direction": "generator",
  "direction_confidence": 0.78,
  "vane_ranking": [[3, 0.8912], [7, 0.7654], ...],
  "detector_votes": { "ocsvm": false, "isolation_forest": false },
  "trained": true
}
```

If models are not trained, `trained` is `false` and `message` explains the limitation.

---

## Training

### `POST /api/train`

Train detection models on simulated normal data.

**Request body:**
```json
{
  "duration_s": 300.0
}
```

**Response:**
```json
{
  "status": "trained",
  "n_segments": 30,
  "ensemble_metrics": { ... },
  "state_classifier_metrics": { ... }
}
```

### `GET /api/train/status`

Check training status.

**Response:**
```json
{
  "status": "trained",
  "is_trained": true,
  "n_segments": 30
}
```

Possible `status` values: `not_trained`, `training`, `trained`, `failed`.

---

## Batch Detection

### `POST /api/detect-all`

Run detection on all loaded segments.

**Response:**
```json
{
  "n_segments": 6,
  "n_anomalies": 2,
  "results": [ ... ]
}
```

Each entry in `results` has the same shape as `/api/segments/{seg_id}/detection`.

---

## Alerts

### `GET /api/alerts`

Get all generated alerts.

**Response:** Array of alert objects.
```json
[
  {
    "level": "WARNING",
    "time": "2025-01-01T00:01:00",
    "source": "ensemble_detector",
    "message": "Anomaly detected in segment seg_3"
  }
]
```

Possible `level` values: `INFO`, `WARNING`, `CRITICAL`, `EMERGENCY`.
