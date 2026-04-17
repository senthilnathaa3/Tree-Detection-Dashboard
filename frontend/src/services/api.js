/**
 * API service for communicating with the TreeSat FastAPI backend.
 */

const API_BASE = 'http://localhost:8001/api';

// ─── Single File Operations ────────────────────────────────────────────

export async function uploadAndPredict(file, threshold = 0.5) {
    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch(`${API_BASE}/predict?threshold=${threshold}`, {
        method: 'POST',
        body: formData,
    });

    if (!response.ok) {
        const error = await response.json().catch(() => ({ detail: 'Unknown error' }));
        throw new Error(error.detail || 'Prediction failed');
    }

    return response.json();
}

export async function getPreview(file, mode = 'rgb') {
    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch(`${API_BASE}/preview?mode=${mode}`, {
        method: 'POST',
        body: formData,
    });

    if (!response.ok) {
        throw new Error('Preview generation failed');
    }

    const blob = await response.blob();
    return URL.createObjectURL(blob);
}

// ─── Server Folder Browser ─────────────────────────────────────────────

export async function browseDirectory(path = '/') {
    const response = await fetch(`${API_BASE}/browse?path=${encodeURIComponent(path)}`);
    if (!response.ok) {
        const error = await response.json().catch(() => ({ detail: 'Unknown error' }));
        throw new Error(error.detail || 'Failed to browse directory');
    }
    return response.json();
}

// ─── Dataset Geographic Bounds ─────────────────────────────────────────

export async function getDatasetBounds(datasetPath) {
    const response = await fetch(`${API_BASE}/dataset-bounds?dataset_path=${encodeURIComponent(datasetPath)}`);
    if (!response.ok) {
        const error = await response.json().catch(() => ({ detail: 'Unknown error' }));
        throw new Error(error.detail || 'Failed to get dataset bounds');
    }
    return response.json();
}

// ─── Dataset Analysis ──────────────────────────────────────────────────

export async function analyzeDataset(datasetPath, threshold = 0.5, batchSize = 32) {
    const response = await fetch(`${API_BASE}/analyze-dataset`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            dataset_path: datasetPath,
            threshold: threshold,
            batch_size: batchSize,
        }),
    });

    if (!response.ok) {
        const error = await response.json().catch(() => ({ detail: 'Unknown error' }));
        throw new Error(error.detail || 'Analysis failed');
    }

    return response.json();
}

export async function getAnalysisStatus() {
    const response = await fetch(`${API_BASE}/analysis-status`);
    if (!response.ok) throw new Error('Failed to get analysis status');
    return response.json();
}

export async function analyzeAOI(payload) {
    const response = await fetch(`${API_BASE}/analyze-aoi`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
    });

    if (!response.ok) {
        const error = await response.json().catch(() => ({ detail: 'Unknown error' }));
        throw new Error(error.detail || 'AOI analysis failed');
    }

    return response.json();
}

export async function validateAOIFIA(payload) {
    const response = await fetch(`${API_BASE}/validate-aoi-fia`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
    });

    if (!response.ok) {
        const error = await response.json().catch(() => ({ detail: 'Unknown error' }));
        throw new Error(error.detail || 'AOI FIA validation failed');
    }

    return response.json();
}

export async function evaluateOffline(payload) {
    const response = await fetch(`${API_BASE}/evaluate-offline`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
    });

    if (!response.ok) {
        const error = await response.json().catch(() => ({ detail: 'Unknown error' }));
        throw new Error(error.detail || 'Offline evaluation failed');
    }

    return response.json();
}

export async function validateLocation(payload) {
    const response = await fetch(`${API_BASE}/validate-location`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
    });

    if (!response.ok) {
        const error = await response.json().catch(() => ({ detail: 'Unknown error' }));
        throw new Error(error.detail || 'Location validation failed');
    }

    return response.json();
}

export async function fitRegionalCalibration(payload) {
    const response = await fetch(`${API_BASE}/fit-fia-calibration-regional`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
    });

    if (!response.ok) {
        const error = await response.json().catch(() => ({ detail: 'Unknown error' }));
        throw new Error(error.detail || 'Regional calibration fit failed');
    }
    return response.json();
}

export async function detectCrowns(file, ndviThreshold = 0.45, minAreaPx = 12, modelTreeCount = null) {
    const formData = new FormData();
    formData.append('file', file);
    
    let url = `${API_BASE}/detect-crowns?ndvi_threshold=${encodeURIComponent(ndviThreshold)}&min_area_px=${encodeURIComponent(minAreaPx)}`;
    if (modelTreeCount !== null) {
        url += `&model_tree_count=${encodeURIComponent(modelTreeCount)}`;
    }

    const response = await fetch(url, {
        method: 'POST',
        body: formData,
    });
    if (!response.ok) {
        const error = await response.json().catch(() => ({ detail: 'Unknown error' }));
        throw new Error(error.detail || 'Crown detection failed');
    }
    return response.json();
}

export async function fetchRemoteGeoTiff(payload) {
    const response = await fetch(`${API_BASE}/fetch-remote-geotiff`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
    });
    if (!response.ok) {
        const error = await response.json().catch(() => ({ detail: 'Unknown error' }));
        throw new Error(error.detail || 'Remote GeoTIFF fetch failed');
    }

    const blob = await response.blob();
    const cd = response.headers.get('content-disposition') || '';
    const match = cd.match(/filename="?([^"]+)"?/i);
    const filename = match?.[1] || 'remote_patch.tif';
    return new File([blob], filename, { type: 'image/tiff' });
}

// ─── Chart Data Endpoints ──────────────────────────────────────────────

export async function getSpeciesDistribution() {
    const response = await fetch(`${API_BASE}/species-distribution`);
    if (!response.ok) throw new Error('Failed to get species distribution');
    return response.json();
}

export async function getDensityMap() {
    const response = await fetch(`${API_BASE}/density-map`);
    if (!response.ok) throw new Error('Failed to get density map');
    return response.json();
}

export async function getBiodiversity() {
    const response = await fetch(`${API_BASE}/biodiversity`);
    if (!response.ok) throw new Error('Failed to get biodiversity metrics');
    return response.json();
}

// ─── Existing Endpoints ────────────────────────────────────────────────

export async function getBatchStats() {
    const response = await fetch(`${API_BASE}/batch-stats`);
    if (!response.ok) throw new Error('Failed to get batch stats');
    return response.json();
}

export async function clearBatch() {
    const response = await fetch(`${API_BASE}/batch-clear`, { method: 'DELETE' });
    if (!response.ok) throw new Error('Failed to clear batch');
    return response.json();
}

export async function healthCheck() {
    const response = await fetch(`${API_BASE}/health`);
    return response.json();
}

export async function getSpeciesLabels() {
    const response = await fetch(`${API_BASE}/species-labels`);
    return response.json();
}
