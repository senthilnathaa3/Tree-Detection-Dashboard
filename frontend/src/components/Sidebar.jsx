import { useState, useCallback } from 'react';
import FolderBrowser from './FolderBrowser';

export default function Sidebar({
    onFileSelect,
    file,
    previewUrl,
    previewMode,
    onPreviewModeChange,
    threshold,
    onThresholdChange,
    onPredict,
    loading,
    metadata,
    apiStatus,
    // New dataset props
    datasetPath,
    onDatasetPathChange,
    onAnalyzeDataset,
    analysisStatus,
    analysisProgress,
    analysisTotal,
    inputMode,
    onInputModeChange,
}) {
    const [dragActive, setDragActive] = useState(false);
    const [folderBrowserOpen, setFolderBrowserOpen] = useState(false);

    const handleDrag = useCallback((e) => {
        e.preventDefault();
        e.stopPropagation();
        if (e.type === 'dragenter' || e.type === 'dragover') {
            setDragActive(true);
        } else if (e.type === 'dragleave') {
            setDragActive(false);
        }
    }, []);

    const handleDrop = useCallback((e) => {
        e.preventDefault();
        e.stopPropagation();
        setDragActive(false);
        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            onFileSelect(e.dataTransfer.files[0]);
        }
    }, [onFileSelect]);

    const handleFileInput = useCallback((e) => {
        if (e.target.files && e.target.files[0]) {
            onFileSelect(e.target.files[0]);
        }
    }, [onFileSelect]);

    const progressPercent = analysisTotal > 0
        ? Math.round((analysisProgress / analysisTotal) * 100)
        : 0;

    const mode = inputMode || 'dataset';

    return (
        <aside className="sidebar">
            {/* Input Mode Toggle */}
            <div className="sidebar-section">
                <div className="sidebar-section-title">Input Mode</div>
                <div className="preview-toggle" style={{ marginTop: 0 }}>
                    <button
                        className={`preview-toggle-btn ${mode === 'dataset' ? 'active' : ''}`}
                        onClick={() => onInputModeChange('dataset')}
                    >
                        📁 Dataset Folder
                    </button>
                    <button
                        className={`preview-toggle-btn ${mode === 'single' ? 'active' : ''}`}
                        onClick={() => onInputModeChange('single')}
                    >
                        📄 Single File
                    </button>
                </div>
            </div>

            {/* ─── Dataset Folder Mode ─── */}
            {mode === 'dataset' && (
                <div className="sidebar-section">
                    <div className="sidebar-section-title">Dataset Loader</div>

                    {/* Dataset Path Input with Browse Button */}
                    <div className="dataset-path-container">
                        <label className="dataset-path-label">
                            <span className="dataset-path-icon">📂</span>
                            Dataset Directory
                        </label>
                        <div className="dataset-path-row">
                            <input
                                id="dataset-path-input"
                                type="text"
                                className="dataset-path-input"
                                placeholder="/path/to/dataset/"
                                value={datasetPath || ''}
                                onChange={(e) => onDatasetPathChange(e.target.value)}
                            />
                            <button
                                className="btn-browse"
                                onClick={() => setFolderBrowserOpen(true)}
                                title="Browse server directories"
                            >
                                📂 Browse
                            </button>
                        </div>
                        <div className="dataset-path-hint">
                            Must contain <code>s1/</code> and <code>s2/</code> subdirectories with matching .tif files
                        </div>
                    </div>

                    {/* Dataset Structure Info */}
                    {datasetPath && (
                        <div className="file-info animate-fade-in" style={{ marginTop: 12 }}>
                            <div className="file-info-row">
                                <span className="file-info-label">Structure</span>
                                <span className="file-info-value">s1/ + s2/</span>
                            </div>
                            <div className="file-info-row">
                                <span className="file-info-label">Format</span>
                                <span className="file-info-value">Paired .tif</span>
                            </div>
                            <div className="file-info-row">
                                <span className="file-info-label">Input</span>
                                <span className="file-info-value">S2(13) + S1(2)</span>
                            </div>
                        </div>
                    )}

                    {/* Progress Indicator */}
                    {analysisStatus === 'running' && (
                        <div className="analysis-progress animate-fade-in" style={{ marginTop: 16 }}>
                            <div className="progress-header">
                                <span className="progress-label">
                                    <span className="spinner" style={{ width: 12, height: 12, borderWidth: 2, display: 'inline-block', verticalAlign: 'middle', marginRight: 8 }}></span>
                                    Processing images...
                                </span>
                                <span className="progress-count">{analysisProgress} / {analysisTotal}</span>
                            </div>
                            <div className="progress-bar" style={{ height: 6, marginTop: 8 }}>
                                <div
                                    className="progress-fill"
                                    style={{ width: `${progressPercent}%` }}
                                ></div>
                            </div>
                            <div style={{ fontSize: '0.7rem', color: '#64748b', marginTop: 4, textAlign: 'right' }}>
                                {progressPercent}% complete
                            </div>
                        </div>
                    )}

                    {/* Analysis Completed */}
                    {analysisStatus === 'completed' && (
                        <div className="analysis-complete animate-fade-in" style={{ marginTop: 12 }}>
                            <div style={{
                                background: 'rgba(16, 185, 129, 0.1)',
                                border: '1px solid rgba(16, 185, 129, 0.3)',
                                borderRadius: 10,
                                padding: '10px 14px',
                                fontSize: '0.8rem',
                                color: '#34d399',
                                display: 'flex',
                                alignItems: 'center',
                                gap: 8
                            }}>
                                <span style={{ fontSize: '1.1rem' }}>✅</span>
                                Analysis complete — {analysisProgress} images processed
                            </div>
                        </div>
                    )}

                    {/* Analysis Error */}
                    {analysisStatus === 'error' && (
                        <div className="analysis-error animate-fade-in" style={{ marginTop: 12 }}>
                            <div style={{
                                background: 'rgba(244, 63, 94, 0.1)',
                                border: '1px solid rgba(244, 63, 94, 0.3)',
                                borderRadius: 10,
                                padding: '10px 14px',
                                fontSize: '0.8rem',
                                color: '#f43f5e',
                                display: 'flex',
                                alignItems: 'center',
                                gap: 8
                            }}>
                                <span style={{ fontSize: '1.1rem' }}>❌</span>
                                Analysis failed. Check the dataset path.
                            </div>
                        </div>
                    )}
                </div>
            )}

            {/* ─── Single File Mode ─── */}
            {mode === 'single' && (
                <div className="sidebar-section">
                    <div className="sidebar-section-title">Single File Upload</div>

                    <div
                        className={`dropzone ${dragActive ? 'active' : ''}`}
                        onDragEnter={handleDrag}
                        onDragLeave={handleDrag}
                        onDragOver={handleDrag}
                        onDrop={handleDrop}
                        onClick={() => document.getElementById('file-input').click()}
                    >
                        <div className="dropzone-icon">🛰️</div>
                        <div className="dropzone-text">
                            {file ? file.name : 'Drop .tif file here'}
                        </div>
                        <div className="dropzone-hint">
                            {file ? `${(file.size / 1024).toFixed(1)} KB` : 'Sentinel-1 & S2 200m patches'}
                        </div>
                        <input
                            id="file-input"
                            type="file"
                            accept=".tif,.tiff"
                            onChange={handleFileInput}
                            style={{ display: 'none' }}
                        />
                    </div>

                    {file && metadata && (
                        <div className="file-info animate-fade-in">
                            <div className="file-info-row">
                                <span className="file-info-label">Bands</span>
                                <span className="file-info-value">{metadata.bands || 15}</span>
                            </div>
                            <div className="file-info-row">
                                <span className="file-info-label">Resolution</span>
                                <span className="file-info-value">200m</span>
                            </div>
                            <div className="file-info-row">
                                <span className="file-info-label">Size</span>
                                <span className="file-info-value">
                                    {metadata.width || '–'}×{metadata.height || '–'}
                                </span>
                            </div>
                            <div className="file-info-row">
                                <span className="file-info-label">CRS</span>
                                <span className="file-info-value">{metadata.crs || '–'}</span>
                            </div>
                            <div className="file-info-row">
                                <span className="file-info-label">Data Type</span>
                                <span className="file-info-value">{metadata.dtype || '–'}</span>
                            </div>
                        </div>
                    )}
                </div>
            )}

            {/* Preview Section (single file only) */}
            {mode === 'single' && previewUrl && (
                <div className="sidebar-section animate-fade-in">
                    <div className="sidebar-section-title">Image Preview</div>
                    <div className="preview-container">
                        <img src={previewUrl} alt="Preview" className="preview-image" />
                    </div>
                    <div className="preview-toggle">
                        <button
                            className={`preview-toggle-btn ${previewMode === 'rgb' ? 'active' : ''}`}
                            onClick={() => onPreviewModeChange('rgb')}
                        >
                            🔴 RGB
                        </button>
                        <button
                            className={`preview-toggle-btn ${previewMode === 'ndvi' ? 'active' : ''}`}
                            onClick={() => onPreviewModeChange('ndvi')}
                        >
                            🌿 NDVI
                        </button>
                    </div>
                </div>
            )}

            {/* Threshold Slider */}
            {/* <div className="sidebar-section">
                <div className="sidebar-section-title">Model Settings</div>
                <div className="slider-container">
                    <div className="slider-label">
                        <span>Species Threshold</span>
                        <span className="slider-value">{threshold.toFixed(2)}</span>
                    </div>
                    <input
                        type="range"
                        min="0"
                        max="1"
                        step="0.05"
                        value={threshold}
                        onChange={(e) => onThresholdChange(parseFloat(e.target.value))}
                    />
                </div>
            </div> */}

            {/* Run Inference / Analyze Button */}
            <div className="sidebar-section">
                {mode === 'dataset' ? (
                    <button
                        className="btn btn-primary btn-full"
                        onClick={onAnalyzeDataset}
                        disabled={!datasetPath || analysisStatus === 'running'}
                    >
                        {analysisStatus === 'running' ? (
                            <>
                                <span className="spinner" style={{ width: 16, height: 16, borderWidth: 2 }}></span>
                                Analyzing {progressPercent}%...
                            </>
                        ) : (
                            <>🧠 Analyze Dataset</>
                        )}
                    </button>
                ) : (
                    <button
                        className="btn btn-primary btn-full"
                        onClick={onPredict}
                        disabled={!file || loading}
                    >
                        {loading ? (
                            <>
                                <span className="spinner" style={{ width: 16, height: 16, borderWidth: 2 }}></span>
                                Running Inference...
                            </>
                        ) : (
                            <>🧠 Analyze File</>
                        )}
                    </button>
                )}
            </div>

            {/* Model Info */}
            {/* <div className="sidebar-section">
                <div className="sidebar-section-title">Model Info</div>
                <div className="file-info">
                    <div className="file-info-row">
                        <span className="file-info-label">Architecture</span>
                        <span className="file-info-value">ResNet18</span>
                    </div>
                    <div className="file-info-row">
                        <span className="file-info-label">Inputs</span>
                        <span className="file-info-value">15ch (S2+S1)</span>
                    </div>
                    <div className="file-info-row">
                        <span className="file-info-label">Heads</span>
                        <span className="file-info-value">Density + Species</span>
                    </div>
                    <div className="file-info-row">
                        <span className="file-info-label">Species</span>
                        <span className="file-info-value">20 classes</span>
                    </div>
                    <div className="file-info-row">
                        <span className="file-info-label">API</span>
                        <span className="file-info-value" style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                            <span className={`status-dot ${apiStatus === 'online' ? '' : 'offline'}`}></span>
                            {apiStatus}nlinenline
                        </span>
                    </div>
                </div>
            </div> */}

            {/* Folder Browser Modal */}
            <FolderBrowser
                isOpen={folderBrowserOpen}
                onClose={() => setFolderBrowserOpen(false)}
                onSelect={(path) => onDatasetPathChange(path)}
            />
        </aside>
    );
}
