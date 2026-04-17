import { useState, useEffect, useCallback, useRef } from 'react';
import Sidebar from './components/Sidebar';
import KPICards from './components/KPICards';
import TreeAnalytics from './components/TreeAnalytics';
import SpeciesAnalytics from './components/SpeciesAnalytics';
import BiodiversityMetrics from './components/BiodiversityMetrics';
import SpatialHeatmaps from './components/SpatialHeatmaps';
import DatasetMap from './components/DatasetMap';
import {
  uploadAndPredict,
  getPreview,
  healthCheck,
  analyzeDataset,
  getAnalysisStatus,
  getBatchStats,
} from './services/api';

function App() {
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [previewMode, setPreviewMode] = useState('rgb');
  const [threshold, setThreshold] = useState(0.5);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [metadata, setMetadata] = useState(null);
  const [apiStatus, setApiStatus] = useState('checking');
  const [error, setError] = useState(null);

  // Dataset analysis state
  const [inputMode, setInputMode] = useState('dataset');
  const [datasetPath, setDatasetPath] = useState('');
  const [analysisStatus, setAnalysisStatus] = useState('idle');
  const [analysisProgress, setAnalysisProgress] = useState(0);
  const [analysisTotal, setAnalysisTotal] = useState(0);
  const pollRef = useRef(null);

  // Check API health on mount
  useEffect(() => {
    const checkHealth = async () => {
      try {
        await healthCheck();
        setApiStatus('online');
      } catch {
        setApiStatus('offline');
      }
    };
    checkHealth();
    const interval = setInterval(checkHealth, 30000);
    return () => clearInterval(interval);
  }, []);

  // Cleanup polling on unmount
  useEffect(() => {
    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
    };
  }, []);

  // Handle file selection (single file mode)
  const handleFileSelect = useCallback(async (selectedFile) => {
    const ext = selectedFile.name.split('.').pop().toLowerCase();
    if (!['tif', 'tiff'].includes(ext)) {
      setError('Only .tif files are accepted');
      return;
    }

    setFile(selectedFile);
    setError(null);
    setResults(null);

    try {
      const url = await getPreview(selectedFile, 'rgb');
      setPreviewUrl(url);
    } catch {
      console.warn('Preview generation failed');
    }
  }, []);

  // Preview mode toggle
  const handlePreviewModeChange = useCallback(async (mode) => {
    setPreviewMode(mode);
    if (file) {
      try {
        const url = await getPreview(file, mode);
        setPreviewUrl(url);
      } catch {
        console.warn('Preview update failed');
      }
    }
  }, [file]);

  // Run single file prediction
  const handlePredict = useCallback(async () => {
    if (!file) return;

    setLoading(true);
    setError(null);

    try {
      const data = await uploadAndPredict(file, threshold);
      setResults(data);
      setMetadata(data.metadata || null);
    } catch (err) {
      setError(err.message || 'Inference failed');
    } finally {
      setLoading(false);
    }
  }, [file, threshold]);

  // ─── Dataset Analysis Flow ───────────────────────────────────────

  const handleAnalyzeDataset = useCallback(async () => {
    if (!datasetPath) return;

    setError(null);
    setResults(null);
    setAnalysisStatus('running');
    setAnalysisProgress(0);
    setAnalysisTotal(0);

    try {
      await analyzeDataset(datasetPath, threshold);

      // Start polling for progress
      pollRef.current = setInterval(async () => {
        try {
          const status = await getAnalysisStatus();
          setAnalysisProgress(status.progress || 0);
          setAnalysisTotal(status.total || 0);

          if (status.status === 'completed') {
            clearInterval(pollRef.current);
            pollRef.current = null;
            setAnalysisStatus('completed');

            // Transform analysis results to match existing dashboard format
            if (status.results) {
              const analysisResults = transformAnalysisResults(status.results);
              setResults(analysisResults);
            }
          } else if (status.status === 'error') {
            clearInterval(pollRef.current);
            pollRef.current = null;
            setAnalysisStatus('error');
            setError(status.error || 'Analysis failed');
          }
        } catch (err) {
          console.warn('Polling error:', err);
        }
      }, 2000);

    } catch (err) {
      setAnalysisStatus('error');
      setError(err.message || 'Failed to start analysis');
    }
  }, [datasetPath, threshold]);

  // Transform batch analysis results to match the existing single-file result format
  function transformAnalysisResults(analysisData) {
    const ds = analysisData.density_statistics || {};
    const ss = analysisData.species_summary || {};
    const ba = analysisData.biodiversity_aggregate || {};

    const density = ds.mean || 0;
    const patch_area_hectares = 4;
    const max_trees_per_hectare = 1000;
    const trees_per_hectare = density * max_trees_per_hectare;

    const speciesSummary = ss.species_summary || [];
    const species_distribution = speciesSummary.map((sp) => ({
      species: sp.species,
      probability: sp.mean_probability,
      detected: sp.mean_probability >= threshold,
    }));

    if (species_distribution.length === 0 && analysisData.dominant_species_distribution) {
      const dist = analysisData.dominant_species_distribution;
      const total = Object.values(dist).reduce((a, b) => a + b, 0);
      for (const [species, count] of Object.entries(dist)) {
        species_distribution.push({
          species,
          probability: count / total,
          detected: true,
        });
      }
      species_distribution.sort((a, b) => b.probability - a.probability);
    }

    const dominant = species_distribution[0] || { species: 'Unknown', probability: 0 };
    const least = species_distribution[species_distribution.length - 1] || { species: 'Unknown', probability: 0 };

    const allProbs = species_distribution.map((s) => s.probability);
    const mean = allProbs.length > 0 ? allProbs.reduce((a, b) => a + b, 0) / allProbs.length : 0;
    const max = allProbs.length > 0 ? Math.max(...allProbs) : 0;
    const min = allProbs.length > 0 ? Math.min(...allProbs) : 0;
    const stdDev = allProbs.length > 0
      ? Math.sqrt(allProbs.reduce((sum, p) => sum + (p - mean) ** 2, 0) / allProbs.length)
      : 0;

    return {
      total_images: analysisData.total_images || 0,
      avg_tree_count: analysisData.avg_tree_count || 0,
      is_batch: true,
      density,
      tree_count: analysisData.avg_tree_count || round(density * max_trees_per_hectare * patch_area_hectares, 1),
      trees_per_hectare: round(trees_per_hectare, 1),
      trees_per_sqkm: round(trees_per_hectare * 100, 1),
      dominant_species: dominant.species,
      least_species: least.species,
      total_species_detected: species_distribution.filter((s) => s.detected).length,
      species_distribution,
      confidence_stats: {
        average: mean,
        minimum: min,
        maximum: max,
        std_dev: stdDev,
      },
      biodiversity_metrics: {
        species_richness: ba.mean_richness || 0,
        shannon_index: ba.mean_shannon_index || analysisData.biodiversity_index || 0,
        evenness: ba.mean_evenness || 0,
        simpsons_index: ba.simpsons_index || 0,
        biodiversity_score: ba.mean_biodiversity_score || 0,
      },
      patch_area_hectares,
      density_statistics: ds,
      species_summary: ss,
      biodiversity_aggregate: ba,
      dominant_species_distribution: analysisData.dominant_species_distribution || {},
    };
  }

  function round(val, decimals) {
    return Math.round(val * 10 ** decimals) / 10 ** decimals;
  }

  const isAnalysisRunning = analysisStatus === 'running';

  return (
    <div className="app-layout">
      {/* Header */}
      <header className="app-header">
        <div className="header-brand">
          <div className="header-logo">🌍</div>
          <div>
            <div className="header-title">Satellite Tree Density & Species Analytics Dashboard</div>
          </div>
        </div>
        <a
          href="/location-validation"
          style={{
            color: '#052e25',
            background: '#34d399',
            textDecoration: 'none',
            fontWeight: 700,
            padding: '8px 12px',
            borderRadius: 10,
            fontSize: '0.78rem',
            border: '1px solid rgba(16,185,129,0.45)',
          }}
        >
          Lat/Lon Validation
        </a>
      </header>

      {/* Sidebar */}
      <Sidebar
        onFileSelect={handleFileSelect}
        file={file}
        previewUrl={previewUrl}
        previewMode={previewMode}
        onPreviewModeChange={handlePreviewModeChange}
        threshold={threshold}
        onThresholdChange={setThreshold}
        onPredict={handlePredict}
        loading={loading}
        metadata={metadata}
        apiStatus={apiStatus}
        datasetPath={datasetPath}
        onDatasetPathChange={setDatasetPath}
        onAnalyzeDataset={handleAnalyzeDataset}
        analysisStatus={analysisStatus}
        analysisProgress={analysisProgress}
        analysisTotal={analysisTotal}
        inputMode={inputMode}
        onInputModeChange={setInputMode}
      />

      {/* Main Content */}
      <main className="main-content">
        {/* Error Banner */}
        {error && (
          <div style={{
            background: 'rgba(244, 63, 94, 0.1)',
            border: '1px solid rgba(244, 63, 94, 0.3)',
            borderRadius: 12,
            padding: '12px 20px',
            marginBottom: 24,
            display: 'flex',
            alignItems: 'center',
            gap: 12,
            color: '#f43f5e',
            fontSize: '0.85rem',
            fontWeight: 500,
            animation: 'fadeInUp 0.3s ease-out',
          }}>
            <span style={{ fontSize: '1.2rem' }}>⚠️</span>
            {error}
            <button
              onClick={() => setError(null)}
              style={{
                marginLeft: 'auto', background: 'none', border: 'none',
                color: '#f43f5e', cursor: 'pointer', fontSize: '1.1rem',
              }}
            >
              ✕
            </button>
          </div>
        )}

        {/* Loading State — single file */}
        {loading && !isAnalysisRunning && (
          <div className="loading-overlay">
            <div className="spinner"></div>
            <div className="loading-text">Running inference on {file?.name}...</div>
            <div style={{ fontSize: '0.75rem', color: '#64748b' }}>
              Processing 15-channel tensor through ResNet18
            </div>
          </div>
        )}

        {/* Loading State — dataset analysis */}
        {isAnalysisRunning && (
          <div className="loading-overlay">
            <div className="spinner"></div>
            <div className="loading-text">
              Analyzing dataset — {analysisProgress} / {analysisTotal} images
            </div>
            <div style={{ width: '100%', maxWidth: 400, marginTop: 12 }}>
              <div className="progress-bar" style={{ height: 8 }}>
                <div
                  className="progress-fill"
                  style={{
                    width: analysisTotal > 0
                      ? `${(analysisProgress / analysisTotal) * 100}%`
                      : '0%'
                  }}
                ></div>
              </div>
            </div>
            <div style={{ fontSize: '0.75rem', color: '#64748b', marginTop: 8 }}>
              Batch inference with ResNet18 · {analysisTotal} paired S1/S2 images
            </div>
          </div>
        )}

        {/* Empty State */}
        {!results && !loading && !isAnalysisRunning && (
          <div className="empty-state">
            <div className="empty-state-icon">🛰️</div>
            <div className="empty-state-title">Satellite Tree Density & Species Analytics Dashboard</div>
            <div className="empty-state-desc">
              {inputMode === 'dataset'
                ? 'Enter a dataset folder path containing s1/ and s2/ subdirectories with paired Sentinel .tif files, then click Analyze Dataset.'
                : 'Upload a Sentinel-1 & Sentinel-2 .tif file (200m resolution) to run deep learning inference and visualize ecological analytics.'
              }
            </div>
            <div style={{
              marginTop: 32,
              display: 'grid',
              gridTemplateColumns: 'repeat(3, 1fr)',
              gap: 16,
              maxWidth: 600,
            }}>
              {[
                { icon: '🌲', title: 'Tree Analytics', desc: 'Count, density, and distribution' },
                { icon: '🌿', title: 'Species Analysis', desc: '20-class multi-label classification' },
                { icon: '🧬', title: 'Biodiversity', desc: 'Shannon index, evenness, richness' },
              ].map((item) => (
                <div key={item.title} style={{
                  background: 'rgba(26, 35, 50, 0.6)',
                  border: '1px solid rgba(148, 163, 184, 0.1)',
                  borderRadius: 12,
                  padding: '20px 16px',
                  textAlign: 'center',
                }}>
                  <div style={{ fontSize: '2rem', marginBottom: 8 }}>{item.icon}</div>
                  <div style={{ fontSize: '0.85rem', fontWeight: 600, color: '#f1f5f9', marginBottom: 4 }}>
                    {item.title}
                  </div>
                  <div style={{ fontSize: '0.75rem', color: '#64748b' }}>
                    {item.desc}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Results Dashboard */}
        {results && !loading && !isAnalysisRunning && (
          <>
            {/* Batch Summary Banner */}
            {results.is_batch && (
              <div style={{
                background: 'linear-gradient(135deg, rgba(5, 150, 105, 0.15), rgba(20, 184, 166, 0.1))',
                border: '1px solid rgba(16, 185, 129, 0.3)',
                borderRadius: 14,
                padding: '16px 24px',
                marginBottom: 24,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                animation: 'fadeInUp 0.3s ease-out',
              }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                  <span style={{ fontSize: '1.4rem' }}>📊</span>
                  <div>
                    <div style={{ fontSize: '0.95rem', fontWeight: 700, color: '#34d399' }}>
                      Batch Analysis Complete
                    </div>
                    <div style={{ fontSize: '0.78rem', color: '#94a3b8' }}>
                      {results.total_images} paired images · Avg {results.avg_tree_count} trees/patch · CSV saved
                    </div>
                  </div>
                </div>
                <div style={{
                  fontFamily: "'JetBrains Mono', monospace",
                  fontSize: '0.75rem',
                  color: '#64748b',
                  background: 'rgba(0,0,0,0.2)',
                  padding: '6px 12px',
                  borderRadius: 8,
                }}>
                  {/* outputs/predictions.csv */}
                </div>
              </div>
            )}

            {/* KPI Summary Row */}
            <KPICards results={results} />

            {/* 🛰 Geographic Coverage Map — works for both modes */}
            {inputMode === 'dataset' && datasetPath && (
              <DatasetMap datasetPath={datasetPath} />
            )}
            {inputMode !== 'dataset' && metadata?.geographic_bounds && (
              <DatasetMap
                boundsData={metadata.geographic_bounds}
                label={`Geographic location of ${file?.name || 'uploaded image'}`}
              />
            )}


            {/* Species Analytics */}
            <SpeciesAnalytics results={results} />

            {/* Biodiversity Metrics */}
            <BiodiversityMetrics results={results} />

            {/* Spatial Heatmaps */}
            <SpatialHeatmaps results={results} />
          </>
        )}
      </main>
    </div>
  );
}

export default App;
