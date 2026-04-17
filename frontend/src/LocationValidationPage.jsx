import { useMemo, useState } from 'react';
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { detectCrowns, fetchRemoteGeoTiff, validateLocation } from './services/api';

const FOREST_PRESET = {
  lat: 38.7,
  lon: -79.9,
  radius_km: 10,
  provider: 'planetary_computer',
  start_date: '2024-06-01',
  end_date: '2024-08-31',
  cloud_cover_max: 30,
  threshold: 0.5,
  validation_source: 'fia',
  fia_csv_path: '',
  worldcover_path: '',
  year_start: 2018,
  year_end: 2024,
  calibration_slope: '',
  calibration_intercept: '',
  calibration_profile_path: '',
  calibration_region: '',
  sample_grid_size: 1,
};

function NumberInput({ label, value, onChange, step = 'any', min, max }) {
  return (
    <label style={styles.field}>
      <span style={styles.label}>{label}</span>
      <input
        style={styles.input}
        type="number"
        value={value}
        step={step}
        min={min}
        max={max}
        onChange={(e) => onChange(e.target.value)}
      />
    </label>
  );
}

function TextInput({ label, value, onChange, placeholder = '' }) {
  return (
    <label style={styles.field}>
      <span style={styles.label}>{label}</span>
      <input
        style={styles.input}
        type="text"
        value={value}
        placeholder={placeholder}
        onChange={(e) => onChange(e.target.value)}
      />
    </label>
  );
}

function DateInput({ label, value, onChange }) {
  return (
    <label style={styles.field}>
      <span style={styles.label}>{label}</span>
      <input style={styles.input} type="date" value={value} onChange={(e) => onChange(e.target.value)} />
    </label>
  );
}

function fmtNum(v, digits = 1) {
  if (v === null || v === undefined || Number.isNaN(Number(v))) return '—';
  return Number(v).toFixed(digits);
}

function accuracyFromPercentDiff(pctDiff) {
  if (pctDiff === null || pctDiff === undefined || Number.isNaN(Number(pctDiff))) return null;
  return Math.max(0, 100 - Math.abs(Number(pctDiff)));
}

export default function LocationValidationPage() {
  const [form, setForm] = useState(FOREST_PRESET);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [result, setResult] = useState(null);
  const [showJson, setShowJson] = useState(false);
  const [isConfigExpanded, setIsConfigExpanded] = useState(true);
  const [crownFile, setCrownFile] = useState(null);
  const [crownThreshold, setCrownThreshold] = useState(0.45);
  const [crownMinArea, setCrownMinArea] = useState(12);
  const [crownFetchRadiusKm, setCrownFetchRadiusKm] = useState(0.2);
  const [crownLoading, setCrownLoading] = useState(false);
  const [crownFetchLoading, setCrownFetchLoading] = useState(false);
  const [crownError, setCrownError] = useState('');
  const [crownResult, setCrownResult] = useState(null);
  const [isDetectionExpanded, setIsDetectionExpanded] = useState(false);
  const [shouldAlignWithModel, setShouldAlignWithModel] = useState(true);

  const isFIA = form.validation_source === 'fia';

  const payload = useMemo(() => {
    const base = {
      lat: Number(form.lat),
      lon: Number(form.lon),
      radius_km: Number(form.radius_km),
      provider: form.provider,
      start_date: form.start_date,
      end_date: form.end_date,
      cloud_cover_max: Number(form.cloud_cover_max),
      threshold: Number(form.threshold),
      validation_source: form.validation_source,
      calibration_slope: form.calibration_slope === '' ? undefined : Number(form.calibration_slope),
      calibration_intercept: form.calibration_intercept === '' ? undefined : Number(form.calibration_intercept),
      calibration_profile_path: form.calibration_profile_path || undefined,
      calibration_region: form.calibration_region || undefined,
      sample_grid_size: Number(form.sample_grid_size || 1),
    };

    if (isFIA) {
      return {
        ...base,
        fia_csv_path: form.fia_csv_path,
        year_start: form.year_start === '' ? undefined : Number(form.year_start),
        year_end: form.year_end === '' ? undefined : Number(form.year_end),
      };
    }

    return {
      ...base,
      worldcover_path: form.worldcover_path,
    };
  }, [form, isFIA]);

  const setField = (key, value) => {
    setForm((prev) => ({ ...prev, [key]: value }));
  };

  const onSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError('');
    setResult(null);
    try {
      const data = await validateLocation(payload);
      setResult(data);
      setIsConfigExpanded(false);
    } catch (err) {
      setError(err.message || 'Location validation failed');
    } finally {
      setLoading(false);
    }
  };

  const runCrownDetection = async (e) => {
    e.preventDefault();
    if (!crownFile) {
      setCrownError('Please select a GeoTIFF file first.');
      return;
    }
    setCrownLoading(true);
    setCrownError('');
    setCrownResult(null);

    // Get target count from model result if available and desired
    let targetCount = null;
    if (shouldAlignWithModel && result) {
      // Use calibrated TPH if available, otherwise raw model TPH
      const tph = insights?.calibratedTPH ?? insights?.modelTPH;
      if (tph) {
          // Calculate area in hectares for the crown TIFF
          // Area of circle = pi * r^2. 1 km^2 = 100 hectares.
          const radiusKm = Number(crownFetchRadiusKm);
          const areaHa = (Math.PI * Math.pow(radiusKm, 2)) * 100;
          targetCount = tph * areaHa;
      }
    }

    try {
      const res = await detectCrowns(
        crownFile, 
        Number(crownThreshold), 
        Number(crownMinArea),
        targetCount
      );
      setCrownResult(res);
    } catch (err) {
      setCrownError(err.message || 'Crown detection failed');
    } finally {
      setCrownLoading(false);
    }
  };

  const autoFetchCrownTiff = async () => {
    setCrownError('');
    setCrownResult(null);
    setCrownFetchLoading(true);
    try {
      const file = await fetchRemoteGeoTiff({
        lat: Number(form.lat),
        lon: Number(form.lon),
        start_date: form.start_date,
        end_date: form.end_date,
        radius_km: Number(crownFetchRadiusKm),
        provider: form.provider,
        cloud_cover_max: Number(form.cloud_cover_max),
      });
      setCrownFile(file);
    } catch (err) {
      setCrownError(err.message || 'Remote GeoTIFF fetch failed');
    } finally {
      setCrownFetchLoading(false);
    }
  };

  const insights = useMemo(() => {
    if (!result) return null;

    const raw = result?.comparison?.density_agreement || {};
    const calibrated = result?.comparison?.density_agreement_calibrated || null;
    const modelSummary = result?.model?.summary || {};
    const prediction = result?.model?.prediction || {};

    const modelTPH = raw.model_mean_trees_per_hectare ?? modelSummary.mean_trees_per_hectare ?? null;
    const fiaTPH = raw.fia_mean_trees_per_hectare ?? result?.external_summary?.summary?.trees_per_hectare?.mean ?? null;
    const calibratedTPH = calibrated?.model_tph_calibrated ?? null;

    const rawPct = raw.percent_difference ?? null;
    const calPct = calibrated?.percent_difference ?? null;

    const rawAccuracy = accuracyFromPercentDiff(rawPct);
    const calibratedAccuracy = accuracyFromPercentDiff(calPct);

    const modelTreeCount = prediction.tree_count ?? modelSummary.avg_tree_count ?? null;
    const fiaTreeCountEstimate = fiaTPH !== null ? Number(fiaTPH) * 4 : null;

    const tphChartData = [
      { metric: 'Model (Raw)', value: modelTPH, fill: '#3b82f6' },
      { metric: 'Model (Calibrated)', value: calibratedTPH, fill: '#a78bfa' },
      { metric: 'FIA', value: fiaTPH, fill: '#10b981' },
    ].filter((d) => d.value !== null && d.value !== undefined);

    const treeCountChartData = [
      { metric: 'Model Tree Count', value: modelTreeCount, fill: '#0ea5e9' },
      { metric: 'FIA Tree Count (4ha est.)', value: fiaTreeCountEstimate, fill: '#22c55e' },
    ].filter((d) => d.value !== null && d.value !== undefined);

    return {
      modelTPH,
      fiaTPH,
      calibratedTPH,
      modelTreeCount,
      fiaTreeCountEstimate,
      rawPct,
      calPct,
      rawAccuracy,
      calibratedAccuracy,
      tphChartData,
      treeCountChartData,
    };
  }, [result]);

  return (
    <div style={styles.page}>
      <header style={styles.header}>
        <div>
          <h1 style={styles.h1}>Lat/Lon Validation</h1>
          <p style={styles.sub}>Remote inference + FIA/ESA benchmark comparison</p>
        </div>
        <div style={{ display: 'flex', gap: 12 }}>
          <button 
            style={styles.linkBtn} 
            onClick={() => setIsConfigExpanded(!isConfigExpanded)}
          >
            {isConfigExpanded ? 'Hide Parameters' : 'Edit Configuration'}
          </button>
          <a href="/" style={{ ...styles.linkBtn, background: 'var(--green-600)', borderColor: 'var(--green-500)', color: 'white' }}>
            Open Dashboard
          </a>
        </div>
      </header>

      <div style={styles.formCard}>
        {isConfigExpanded && (
          <>
            <div style={styles.configHeader}>
              <div style={styles.h2}>Validation Parameters</div>
              <button 
                type="button" 
                style={styles.secondaryBtn} 
                onClick={() => setForm(FOREST_PRESET)}
                disabled={loading}
              >
                Reset Preset
              </button>
            </div>
            <form onSubmit={onSubmit} style={styles.formContent}>
              <div style={styles.grid3}>
                <NumberInput label="Latitude" value={form.lat} onChange={(v) => setField('lat', v)} step="0.0001" min={-90} max={90} />
                <NumberInput label="Longitude" value={form.lon} onChange={(v) => setField('lon', v)} step="0.0001" min={-180} max={180} />
                <NumberInput label="Radius (km)" value={form.radius_km} onChange={(v) => setField('radius_km', v)} step="0.1" min={0.1} />
              </div>

              <div style={styles.grid3}>
                <DateInput label="Start Date" value={form.start_date} onChange={(v) => setField('start_date', v)} />
                <DateInput label="End Date" value={form.end_date} onChange={(v) => setField('end_date', v)} />
                <NumberInput label="Cloud Cover Max (%)" value={form.cloud_cover_max} onChange={(v) => setField('cloud_cover_max', v)} step="1" min={0} max={100} />
              </div>

              <div style={styles.grid3}>
                <NumberInput label="Species Threshold" value={form.threshold} onChange={(v) => setField('threshold', v)} step="0.01" min={0} max={1} />

                <label style={styles.field}>
                  <span style={styles.label}>Provider</span>
                  <select style={styles.input} value={form.provider} onChange={(e) => setField('provider', e.target.value)}>
                    <option value="planetary_computer">Planetary Computer</option>
                  </select>
                </label>

                <label style={styles.field}>
                  <span style={styles.label}>Validation Source</span>
                  <select
                    style={styles.input}
                    value={form.validation_source}
                    onChange={(e) => setField('validation_source', e.target.value)}
                  >
                    <option value="fia">FIA</option>
                    <option value="esa_worldcover">ESA WorldCover</option>
                  </select>
                </label>
              </div>

              <div style={styles.grid2}>
                <NumberInput
                  label="Sample Grid Size (1/3/5)"
                  value={form.sample_grid_size}
                  onChange={(v) => setField('sample_grid_size', v)}
                  step="1"
                  min={1}
                />
                <NumberInput
                  label="Calibration Slope"
                  value={form.calibration_slope}
                  onChange={(v) => setField('calibration_slope', v)}
                  step="0.000001"
                />
                <NumberInput
                  label="Calibration Intercept"
                  value={form.calibration_intercept}
                  onChange={(v) => setField('calibration_intercept', v)}
                  step="0.000001"
                />
              </div>
              
              <div style={styles.grid2}>
                <TextInput
                  label="Calibration Profile Path"
                  value={form.calibration_profile_path}
                  onChange={(v) => setField('calibration_profile_path', v)}
                  placeholder="/abs/path/regional_calibration.json"
                />
                <TextInput
                  label="Calibration Region"
                  value={form.calibration_region}
                  onChange={(v) => setField('calibration_region', v)}
                  placeholder="e.g., appalachia_wv"
                />
              </div>

              {isFIA ? (
                <>
                  <TextInput
                    label="FIA CSV Path"
                    value={form.fia_csv_path}
                    onChange={(v) => setField('fia_csv_path', v)}
                    placeholder="/abs/path/to/fia.csv"
                  />
                  <div style={styles.grid2}>
                    <NumberInput label="Year Start" value={form.year_start} onChange={(v) => setField('year_start', v)} step="1" />
                    <NumberInput label="Year End" value={form.year_end} onChange={(v) => setField('year_end', v)} step="1" />
                  </div>
                </>
              ) : (
                <TextInput
                  label="WorldCover Raster Path"
                  value={form.worldcover_path}
                  onChange={(v) => setField('worldcover_path', v)}
                  placeholder="/abs/path/to/worldcover.tif"
                />
              )}

              <div style={styles.actions}>
                <button type="submit" style={styles.primaryBtn} disabled={loading}>
                  {loading ? 'Processing Validation...' : 'Run Benchmark'}
                </button>
              </div>
            </form>
          </>
        )}
      </div>

      {error ? <div style={styles.error}>{error}</div> : null}

      {result && insights ? (
        <>
          <section style={styles.resultCard}>
            <h2 style={styles.h2}>Pitch Metrics</h2>
            <div style={styles.kpiGrid}>
              <MetricCard
                title="Agreement Accuracy (Raw)"
                value={`${fmtNum(insights.rawAccuracy, 2)}%`}
                subtitle="Proxy = 100 - |percent difference|"
                tone="blue"
              />
              <MetricCard
                title="Agreement Accuracy (Calibrated)"
                value={insights.calibratedAccuracy === null ? '—' : `${fmtNum(insights.calibratedAccuracy, 2)}%`}
                subtitle="Shown when calibration is provided"
                tone="purple"
              />
              <MetricCard
                title="Model TPH (Raw)"
                value={fmtNum(insights.modelTPH, 2)}
                subtitle="Trees per hectare"
                tone="sky"
              />
              <MetricCard
                title="FIA TPH"
                value={fmtNum(insights.fiaTPH, 2)}
                subtitle="FIA AOI reference"
                tone="green"
              />
            </div>
          </section>

          <section style={styles.resultCard}>
            <h2 style={styles.h2}>Model vs FIA Visual Comparison</h2>
            <div style={styles.chartGrid}>
              <div style={styles.chartBox}>
                <div style={styles.chartTitle}>Trees per Hectare (TPH)</div>
                <ResponsiveContainer width="100%" height={260}>
                  <BarChart data={insights.tphChartData} margin={{ top: 12, right: 18, left: 0, bottom: 12 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.25)" />
                    <XAxis dataKey="metric" tick={{ fill: 'var(--text-muted)', fontSize: 11, fontWeight: 500 }} axisLine={{ stroke: 'var(--border-color)' }} tickLine={false} />
                    <YAxis tick={{ fill: 'var(--text-muted)', fontSize: 11, fontWeight: 500 }} axisLine={{ stroke: 'var(--border-color)' }} tickLine={false} />
                    <Tooltip
                      contentStyle={{ backgroundColor: 'var(--bg-elevated)', border: '1px solid var(--border-color)', borderRadius: 10, boxShadow: 'var(--shadow-lg)' }}
                      labelStyle={{ color: 'var(--text-primary)', fontWeight: 600, marginBottom: 4 }}
                      itemStyle={{ fontSize: '0.85rem' }}
                    />
                    <Legend />
                    <Bar dataKey="value" name="TPH" radius={[8, 8, 0, 0]}>
                      {insights.tphChartData.map((entry) => (
                        <Cell key={entry.metric} fill={entry.fill} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>

              <div style={styles.chartBox}>
                <div style={styles.chartTitle}>Tree Count Comparison (4ha basis)</div>
                <ResponsiveContainer width="100%" height={260}>
                  <BarChart data={insights.treeCountChartData} margin={{ top: 12, right: 18, left: 0, bottom: 12 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.25)" />
                    <XAxis dataKey="metric" tick={{ fill: 'var(--text-muted)', fontSize: 11, fontWeight: 500 }} axisLine={{ stroke: 'var(--border-color)' }} tickLine={false} />
                    <YAxis tick={{ fill: 'var(--text-muted)', fontSize: 11, fontWeight: 500 }} axisLine={{ stroke: 'var(--border-color)' }} tickLine={false} />
                    <Tooltip
                      contentStyle={{ backgroundColor: 'var(--bg-elevated)', border: '1px solid var(--border-color)', borderRadius: 10, boxShadow: 'var(--shadow-lg)' }}
                      labelStyle={{ color: 'var(--text-primary)', fontWeight: 600, marginBottom: 4 }}
                      itemStyle={{ fontSize: '0.85rem' }}
                    />
                    <Bar dataKey="value" name="Tree Count" radius={[8, 8, 0, 0]}>
                      {insights.treeCountChartData.map((entry) => (
                        <Cell key={entry.metric} fill={entry.fill} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
                <div style={styles.smallNote}>FIA tree count is estimated from FIA TPH x 4 hectares for side-by-side pitch visualization.</div>
              </div>
            </div>
          </section>

          <section style={styles.resultCard}>
            <h2 style={styles.h2}>Run Summary</h2>
            <div style={styles.summaryGrid}>
              <SummaryRow label="Model status" value={`${result?.model?.status || '—'}${result?.model?.mode ? ` (${result.model.mode})` : ''}`} />
              <SummaryRow label="Provider" value={result?.input?.provider || '—'} />
              <SummaryRow label="AOI Center" value={`${fmtNum(result?.input?.lat, 4)}, ${fmtNum(result?.input?.lon, 4)}`} />
              <SummaryRow label="AOI Radius" value={`${fmtNum(result?.input?.radius_km, 2)} km`} />
              <SummaryRow label="Raw Percent Difference" value={insights.rawPct === null ? '—' : `${fmtNum(insights.rawPct, 2)}%`} />
              <SummaryRow label="Calibrated Percent Difference" value={insights.calPct === null ? '—' : `${fmtNum(insights.calPct, 2)}%`} />
            </div>
          </section>

          <section style={styles.resultCard}>
            <div style={styles.rowBetween}>
              <h2 style={styles.h2}>Raw Response (JSON)</h2>
              <button style={styles.secondaryBtn} type="button" onClick={() => setShowJson((s) => !s)}>
                {showJson ? 'Hide JSON' : 'Show JSON'}
              </button>
            </div>
            {showJson ? <pre style={styles.pre}>{JSON.stringify(result, null, 2)}</pre> : null}
          </section>
        </>
      ) : null}

      <section style={{ ...styles.resultCard, marginTop: 40, background: 'rgba(255,255,255,0.01)' }}>
        <div style={styles.rowBetween}>
          <div>
            <h2 style={{ ...styles.h2, margin: 0 }}>Developer Tool: Object Detection</h2>
            <p style={styles.subtleLine}>NDVI + connected-components baseline for crown candidate detection</p>
          </div>
          <button 
            style={styles.secondaryBtn} 
            type="button" 
            onClick={() => setIsDetectionExpanded(!isDetectionExpanded)}
          >
            {isDetectionExpanded ? 'Hide Detection Tool' : 'Show Detection Tool'}
          </button>
        </div>

        {isDetectionExpanded && (
          <div style={{ marginTop: 24, paddingTop: 24, borderTop: '1px solid var(--border-color)' }}>
            <form onSubmit={runCrownDetection} style={styles.crownForm}>
              <div style={styles.fetchBox}>
                <div style={styles.fetchRow}>
                  <NumberInput
                    label="Fetch Radius (km)"
                    value={crownFetchRadiusKm}
                    onChange={(v) => setCrownFetchRadiusKm(v)}
                    step="0.05"
                    min={0.05}
                  />
                  <div style={{ display: 'flex', alignItems: 'end' }}>
                    <button
                      type="button"
                      style={styles.primaryBtn}
                      onClick={autoFetchCrownTiff}
                      disabled={crownFetchLoading || crownLoading}
                    >
                      {crownFetchLoading ? 'Fetching GeoTIFF...' : 'Auto-Fetch GeoTIFF (Lat/Lon)'}
                    </button>
                  </div>
                </div>
                <div style={styles.smallNote}>
                  Uses current latitude/longitude, date range, provider, and cloud-cover settings from above.
                </div>
              </div>

              <label style={styles.field}>
                <span style={styles.label}>GeoTIFF File</span>
                <input
                  style={styles.fileInput}
                  type="file"
                  accept=".tif,.tiff"
                  onChange={(e) => setCrownFile(e.target.files?.[0] || null)}
                />
                <span style={styles.fileName}>
                  {crownFile ? `Selected: ${crownFile.name}` : 'No file selected yet'}
                </span>
              </label>

          <div style={styles.grid3}>
            <NumberInput
              label="NDVI Threshold"
              value={crownThreshold}
              onChange={(v) => setCrownThreshold(v)}
              step="0.01"
              min={-1}
              max={1}
            />
            <NumberInput
              label="Min Area (px)"
              value={crownMinArea}
              onChange={(v) => setCrownMinArea(v)}
              step="1"
              min={1}
            />
            <label style={{ ...styles.field, cursor: 'pointer', display: 'flex', flexDirection: 'row', alignItems: 'center', gap: 8 }}>
              <input
                type="checkbox"
                checked={shouldAlignWithModel}
                onChange={(e) => setShouldAlignWithModel(e.target.checked)}
                style={{ width: 18, height: 18 }}
              />
              <span style={{ ...styles.label, textTransform: 'none', fontSize: '0.85rem' }}>Align Count with Model</span>
            </label>
          </div>
          <div style={{ display: 'flex', justifyContent: 'flex-end', marginTop: 12 }}>
            <button type="submit" style={{ ...styles.primaryBtn, background: 'var(--blue-600)' }} disabled={crownLoading || crownFetchLoading}>
              {crownLoading ? 'Detecting...' : 'Run Crown Detection'}
            </button>
          </div>
        </form>

            {crownError ? <div style={styles.errorInline}>{crownError}</div> : null}

            {crownResult ? (
              <>
                <div style={styles.kpiGrid}>
                  <MetricCard
                    title="Detected Candidates"
                    value={fmtNum(crownResult.candidate_count, 0)}
                    subtitle="Object-level crown candidates"
                    tone="green"
                  />
                  <MetricCard
                    title="NDVI Threshold"
                    value={fmtNum(crownResult.ndvi_threshold, 2)}
                    subtitle="Detection sensitivity"
                    tone="sky"
                  />
                  <MetricCard
                    title="Min Area"
                    value={`${fmtNum(crownResult.min_area_px, 0)} px`}
                    subtitle="Noise filtering"
                    tone="purple"
                  />
                </div>

                <div style={{ marginTop: 24 }}>
                  <div style={styles.chartTitle}>Top Crown Candidates</div>
                  <div style={styles.tableWrap}>
                    <table style={styles.table}>
                      <thead style={{ background: 'rgba(255,255,255,0.03)' }}>
                        <tr>
                          <th style={{ padding: '12px 14px', textAlign: 'left' }}>#</th>
                          <th style={{ padding: '12px 14px', textAlign: 'left' }}>Score</th>
                          <th style={{ padding: '12px 14px', textAlign: 'left' }}>Area(px)</th>
                          <th style={{ padding: '12px 14px', textAlign: 'left' }}>Centroid(px)</th>
                          <th style={{ padding: '12px 14px', textAlign: 'left' }}>BBox(px)</th>
                        </tr>
                      </thead>
                      <tbody>
                        {(crownResult.detections || []).slice(0, 20).map((d, i) => (
                          <tr key={`${i}-${d.score}`} style={{ borderTop: '1px solid var(--border-color)' }}>
                            <td style={{ padding: '12px 14px' }}>{i + 1}</td>
                            <td style={{ padding: '12px 14px' }}>{fmtNum(d.score, 3)}</td>
                            <td style={{ padding: '12px 14px' }}>{fmtNum(d.area_px, 0)}</td>
                            <td style={{ padding: '12px 14px' }}>{fmtNum(d.centroid_px?.x, 1)}, {fmtNum(d.centroid_px?.y, 1)}</td>
                            <td style={{ padding: '12px 14px' }}>
                              {d.bbox_px?.xmin},{d.bbox_px?.ymin} {' -> '} {d.bbox_px?.xmax},{d.bbox_px?.ymax}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              </>
            ) : null}
          </div>
        )}
      </section>
    </div>
  );
}

function MetricCard({ title, value, subtitle, tone }) {
  const tones = {
    blue: '#3b82f6',
    purple: '#a78bfa',
    green: '#10b981',
    sky: '#0ea5e9',
  };
  const color = tones[tone] || tones.blue;
  return (
    <div style={styles.metricCard}>
      <div style={{ ...styles.metricAccent, background: color }} />
      <div style={styles.metricTitle}>{title}</div>
      <div style={styles.metricValue}>{value}</div>
      <div style={styles.metricSub}>{subtitle}</div>
    </div>
  );
}

function SummaryRow({ label, value }) {
  return (
    <div style={styles.summaryRow}>
      <span style={styles.summaryLabel}>{label}</span>
      <span style={styles.summaryValue}>{value}</span>
    </div>
  );
}

const styles = {
  page: {
    minHeight: '100vh',
    background: 'var(--bg-primary)',
    color: 'var(--text-primary)',
    padding: '32px 24px',
    fontFamily: 'var(--font-sans)',
  },
  header: {
    maxWidth: 1200,
    margin: '0 auto 24px',
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    gap: 16,
  },
  h1: { margin: 0, fontSize: '1.6rem', fontWeight: 700, letterSpacing: '-0.025em' },
  h2: { margin: '0 0 16px 0', fontSize: '1.1rem', fontWeight: 600, color: 'var(--text-primary)' },
  sub: { margin: '4px 0 0 0', color: 'var(--text-muted)', fontSize: '0.9rem' },
  linkBtn: {
    color: 'var(--text-primary)',
    background: 'var(--bg-elevated)',
    border: '1px solid var(--border-color)',
    textDecoration: 'none',
    fontWeight: 600,
    padding: '8px 16px',
    borderRadius: 8,
    fontSize: '0.85rem',
    transition: 'all 0.2s ease',
  },
  configHeader: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: '12px 16px',
    background: 'var(--bg-elevated)',
    borderRadius: '12px 12px 0 0',
    borderBottom: '1px solid var(--border-color)',
  },
  formCard: {
    maxWidth: 1200,
    margin: '0 auto',
    background: 'var(--bg-card)',
    border: '1px solid var(--border-color)',
    borderRadius: 12,
    overflow: 'hidden',
  },
  formContent: {
    padding: 20,
    display: 'grid',
    gap: 16,
  },
  grid2: { display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 16 },
  grid3: { display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: 16 },
  field: { display: 'grid', gap: 6 },
  label: { color: 'var(--text-secondary)', fontSize: '0.75rem', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.05em' },
  input: {
    background: 'var(--bg-input)',
    color: 'var(--text-primary)',
    border: '1px solid var(--border-color)',
    borderRadius: 8,
    padding: '10px 12px',
    outline: 'none',
    fontFamily: 'var(--font-sans)',
    fontSize: '0.9rem',
    transition: 'border-color 0.2s ease',
  },
  actions: { 
    display: 'flex', 
    justifyContent: 'flex-end', 
    gap: 12, 
    marginTop: 8,
    padding: '16px 20px',
    background: 'rgba(255,255,255,0.02)',
    borderTop: '1px solid var(--border-color)',
  },
  secondaryBtn: {
    background: 'transparent',
    color: 'var(--text-secondary)',
    border: '1px solid var(--border-color)',
    borderRadius: 8,
    padding: '8px 16px',
    cursor: 'pointer',
    fontFamily: 'var(--font-sans)',
    fontWeight: 500,
    fontSize: '0.85rem',
  },
  primaryBtn: {
    background: 'var(--green-600)',
    color: 'white',
    border: 'none',
    borderRadius: 8,
    padding: '8px 20px',
    cursor: 'pointer',
    fontWeight: 600,
    fontFamily: 'var(--font-sans)',
    fontSize: '0.85rem',
    boxShadow: '0 1px 2px rgba(0,0,0,0.1)',
  },
  error: {
    maxWidth: 1200,
    margin: '16px auto 0',
    borderRadius: 12,
    border: '1px solid rgba(244,63,94,0.3)',
    background: 'rgba(244,63,94,0.05)',
    color: '#fca5a5',
    padding: 12,
    fontSize: '0.85rem',
  },
  resultCard: {
    maxWidth: 1200,
    margin: '24px auto 0',
    background: 'var(--bg-card)',
    border: '1px solid var(--border-color)',
    borderRadius: 12,
    padding: 24,
  },
  kpiGrid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))',
    gap: 16,
  },
  metricCard: {
    background: 'var(--bg-elevated)',
    border: '1px solid var(--border-color)',
    borderRadius: 12,
    padding: 20,
    position: 'relative',
    overflow: 'hidden',
  },
  metricAccent: {
    position: 'absolute',
    left: 0,
    top: 0,
    bottom: 0,
    width: 4,
  },
  metricTitle: { color: 'var(--text-muted)', fontSize: '0.7rem', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: 8 },
  metricValue: { color: 'var(--text-primary)', fontSize: '1.75rem', fontWeight: 700, letterSpacing: '-0.025em' },
  metricSub: { color: 'var(--text-muted)', fontSize: '0.75rem', marginTop: 4 },
  chartGrid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fit, minmax(480px, 1fr))',
    gap: 20,
    marginTop: 24,
  },
  chartBox: {
    border: '1px solid var(--border-color)',
    borderRadius: 12,
    padding: 20,
    background: 'rgba(255,255,255,0.01)',
  },
  chartTitle: { color: 'var(--text-primary)', fontSize: '0.95rem', marginBottom: 20, fontWeight: 600 },
  smallNote: { color: 'var(--text-muted)', fontSize: '0.75rem', marginTop: 12, fontStyle: 'italic' },
  summaryGrid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))',
    gap: 12,
  },
  summaryRow: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    gap: 8,
    border: '1px solid var(--border-color)',
    borderRadius: 8,
    padding: '10px 14px',
    background: 'rgba(255,255,255,0.01)',
  },
  summaryLabel: { color: 'var(--text-muted)', fontSize: '0.8rem' },
  summaryValue: { color: 'var(--text-primary)', fontSize: '0.85rem', fontWeight: 600 },
  rowBetween: { display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 },
  pre: {
    margin: 0,
    maxHeight: '60vh',
    overflow: 'auto',
    fontSize: '0.8rem',
    lineHeight: 1.6,
    color: 'var(--text-secondary)',
    background: 'var(--bg-input)',
    borderRadius: 8,
    border: '1px solid var(--border-color)',
    padding: 16,
    fontFamily: 'var(--font-mono)',
  },
  subtleLine: {
    margin: '0 0 16px 0',
    color: 'var(--text-muted)',
    fontSize: '0.85rem',
  },
  crownForm: {
    display: 'grid',
    gap: 16,
    marginBottom: 20,
  },
  fetchBox: {
    border: '1px solid var(--border-color)',
    background: 'var(--bg-input)',
    borderRadius: 10,
    padding: 16,
    marginBottom: 16,
  },
  fetchRow: {
    display: 'grid',
    gap: 16,
    gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))',
    alignItems: 'end',
  },
  fileInput: {
    background: 'var(--bg-input)',
    color: 'var(--text-primary)',
    border: '1px solid var(--border-color)',
    borderRadius: 8,
    padding: '8px 12px',
    fontFamily: 'var(--font-sans)',
    fontSize: '0.85rem',
  },
  fileName: {
    color: 'var(--text-muted)',
    fontSize: '0.75rem',
    marginTop: 4,
    display: 'block',
  },
  errorInline: {
    border: '1px solid rgba(244,63,94,0.3)',
    background: 'rgba(244,63,94,0.05)',
    color: '#fca5a5',
    borderRadius: 8,
    padding: 12,
    fontSize: '0.8rem',
    marginTop: 12,
  },
  tableWrap: {
    overflowX: 'auto',
    border: '1px solid var(--border-color)',
    borderRadius: 12,
    background: 'rgba(255,255,255,0.01)',
  },
  table: {
    width: '100%',
    borderCollapse: 'collapse',
    fontSize: '0.8rem',
    color: 'var(--text-primary)',
  },
};
