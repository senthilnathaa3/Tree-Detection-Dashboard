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
import { detectCrowns, validateLocation } from './services/api';

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
  const [crownFile, setCrownFile] = useState(null);
  const [crownThreshold, setCrownThreshold] = useState(0.45);
  const [crownMinArea, setCrownMinArea] = useState(12);
  const [crownLoading, setCrownLoading] = useState(false);
  const [crownError, setCrownError] = useState('');
  const [crownResult, setCrownResult] = useState(null);

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
    try {
      const res = await detectCrowns(crownFile, Number(crownThreshold), Number(crownMinArea));
      setCrownResult(res);
    } catch (err) {
      setCrownError(err.message || 'Crown detection failed');
    } finally {
      setCrownLoading(false);
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
          <p style={styles.sub}>Remote inference + FIA/ESA comparison for pitch-ready benchmarking</p>
        </div>
        <a href="/" style={styles.linkBtn}>Open Dashboard</a>
      </header>

      <form onSubmit={onSubmit} style={styles.formCard}>
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
            label="Calibration Slope (optional)"
            value={form.calibration_slope}
            onChange={(v) => setField('calibration_slope', v)}
            step="0.000001"
          />
          <NumberInput
            label="Calibration Intercept (optional)"
            value={form.calibration_intercept}
            onChange={(v) => setField('calibration_intercept', v)}
            step="0.000001"
          />
        </div>
        <div style={styles.grid2}>
          <TextInput
            label="Calibration Profile Path (optional)"
            value={form.calibration_profile_path}
            onChange={(v) => setField('calibration_profile_path', v)}
            placeholder="/abs/path/regional_calibration.json"
          />
          <TextInput
            label="Calibration Region (optional)"
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
              <NumberInput label="Year Start (optional)" value={form.year_start} onChange={(v) => setField('year_start', v)} step="1" />
              <NumberInput label="Year End (optional)" value={form.year_end} onChange={(v) => setField('year_end', v)} step="1" />
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
          <button type="button" style={styles.secondaryBtn} onClick={() => setForm(FOREST_PRESET)} disabled={loading}>
            Reset Preset
          </button>
          <button type="submit" style={styles.primaryBtn} disabled={loading}>
            {loading ? 'Running...' : 'Run Validation'}
          </button>
        </div>
      </form>

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
                    <XAxis dataKey="metric" tick={{ fill: '#cbd5e1', fontSize: 12 }} />
                    <YAxis tick={{ fill: '#cbd5e1', fontSize: 12 }} />
                    <Tooltip
                      contentStyle={{ backgroundColor: '#0b1220', border: '1px solid #334155', borderRadius: 8 }}
                      labelStyle={{ color: '#e2e8f0' }}
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
                    <XAxis dataKey="metric" tick={{ fill: '#cbd5e1', fontSize: 12 }} />
                    <YAxis tick={{ fill: '#cbd5e1', fontSize: 12 }} />
                    <Tooltip
                      contentStyle={{ backgroundColor: '#0b1220', border: '1px solid #334155', borderRadius: 8 }}
                      labelStyle={{ color: '#e2e8f0' }}
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

      <section style={styles.resultCard}>
        <h2 style={styles.h2}>Object Detection: Crown Candidates</h2>
        <p style={styles.subtleLine}>
          NDVI + connected-components baseline for object-level crown candidate detection.
        </p>

        <form onSubmit={runCrownDetection} style={styles.crownForm}>
          <label style={styles.field}>
            <span style={styles.label}>GeoTIFF File</span>
            <input
              style={styles.fileInput}
              type="file"
              accept=".tif,.tiff"
              onChange={(e) => setCrownFile(e.target.files?.[0] || null)}
            />
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
            <div style={{ display: 'flex', alignItems: 'end' }}>
              <button type="submit" style={styles.primaryBtn} disabled={crownLoading}>
                {crownLoading ? 'Detecting...' : 'Run Crown Detection'}
              </button>
            </div>
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

            <div style={{ marginTop: 12 }}>
              <div style={styles.chartTitle}>Top Crown Candidates</div>
              <div style={styles.tableWrap}>
                <table style={styles.table}>
                  <thead>
                    <tr>
                      <th>#</th>
                      <th>Score</th>
                      <th>Area(px)</th>
                      <th>Centroid(px)</th>
                      <th>BBox(px)</th>
                    </tr>
                  </thead>
                  <tbody>
                    {(crownResult.detections || []).slice(0, 20).map((d, i) => (
                      <tr key={`${i}-${d.score}`}>
                        <td>{i + 1}</td>
                        <td>{fmtNum(d.score, 3)}</td>
                        <td>{fmtNum(d.area_px, 0)}</td>
                        <td>{fmtNum(d.centroid_px?.x, 1)}, {fmtNum(d.centroid_px?.y, 1)}</td>
                        <td>
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
      </section>
    </div>
  );
}

function MetricCard({ title, value, subtitle, tone }) {
  const tones = {
    blue: { border: 'rgba(59,130,246,0.4)', bg: 'rgba(59,130,246,0.12)' },
    purple: { border: 'rgba(167,139,250,0.45)', bg: 'rgba(167,139,250,0.12)' },
    green: { border: 'rgba(34,197,94,0.45)', bg: 'rgba(34,197,94,0.12)' },
    sky: { border: 'rgba(14,165,233,0.45)', bg: 'rgba(14,165,233,0.12)' },
  };
  const t = tones[tone] || tones.blue;
  return (
    <div style={{ ...styles.metricCard, borderColor: t.border, background: t.bg }}>
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
    background: 'linear-gradient(180deg, #04151f 0%, #0d1b2a 100%)',
    color: '#e2e8f0',
    padding: '24px',
    fontFamily: "'JetBrains Mono', monospace",
  },
  header: {
    maxWidth: 1260,
    margin: '0 auto 16px',
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
    gap: 16,
  },
  h1: { margin: 0, fontSize: '1.4rem' },
  h2: { margin: '0 0 12px 0', fontSize: '1.05rem' },
  sub: { margin: '8px 0 0 0', color: '#94a3b8', fontSize: '0.9rem' },
  linkBtn: {
    color: '#0f172a',
    background: '#34d399',
    textDecoration: 'none',
    fontWeight: 700,
    padding: '10px 14px',
    borderRadius: 10,
    fontSize: '0.85rem',
  },
  formCard: {
    maxWidth: 1260,
    margin: '0 auto',
    background: 'rgba(15, 23, 42, 0.72)',
    border: '1px solid rgba(148, 163, 184, 0.2)',
    borderRadius: 14,
    padding: 16,
    display: 'grid',
    gap: 12,
  },
  grid2: { display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))', gap: 12 },
  grid3: { display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: 12 },
  field: { display: 'grid', gap: 6 },
  label: { color: '#94a3b8', fontSize: '0.8rem' },
  input: {
    background: '#0f172a',
    color: '#e2e8f0',
    border: '1px solid #334155',
    borderRadius: 10,
    padding: '10px 12px',
    outline: 'none',
    fontFamily: "'JetBrains Mono', monospace",
  },
  actions: { display: 'flex', justifyContent: 'flex-end', gap: 10, marginTop: 6 },
  secondaryBtn: {
    background: 'transparent',
    color: '#cbd5e1',
    border: '1px solid #475569',
    borderRadius: 10,
    padding: '10px 12px',
    cursor: 'pointer',
    fontFamily: "'JetBrains Mono', monospace",
  },
  primaryBtn: {
    background: '#10b981',
    color: '#06231b',
    border: 'none',
    borderRadius: 10,
    padding: '10px 14px',
    cursor: 'pointer',
    fontWeight: 700,
    fontFamily: "'JetBrains Mono', monospace",
  },
  error: {
    maxWidth: 1260,
    margin: '14px auto 0',
    borderRadius: 10,
    border: '1px solid rgba(244,63,94,0.45)',
    background: 'rgba(244,63,94,0.12)',
    color: '#fecdd3',
    padding: 12,
    fontSize: '0.85rem',
  },
  resultCard: {
    maxWidth: 1260,
    margin: '14px auto 0',
    background: 'rgba(15, 23, 42, 0.72)',
    border: '1px solid rgba(148, 163, 184, 0.2)',
    borderRadius: 14,
    padding: 16,
  },
  kpiGrid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))',
    gap: 12,
  },
  metricCard: {
    border: '1px solid',
    borderRadius: 12,
    padding: 12,
  },
  metricTitle: { color: '#cbd5e1', fontSize: '0.75rem', marginBottom: 6 },
  metricValue: { color: '#f8fafc', fontSize: '1.25rem', fontWeight: 700 },
  metricSub: { color: '#94a3b8', fontSize: '0.7rem', marginTop: 4 },
  chartGrid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fit, minmax(420px, 1fr))',
    gap: 12,
  },
  chartBox: {
    border: '1px solid rgba(148,163,184,0.22)',
    borderRadius: 12,
    padding: 10,
    background: 'rgba(2,6,23,0.45)',
  },
  chartTitle: { color: '#e2e8f0', fontSize: '0.85rem', marginBottom: 6, fontWeight: 600 },
  smallNote: { color: '#94a3b8', fontSize: '0.7rem', marginTop: 8 },
  summaryGrid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))',
    gap: 8,
  },
  summaryRow: {
    display: 'flex',
    justifyContent: 'space-between',
    gap: 8,
    border: '1px solid rgba(148,163,184,0.2)',
    borderRadius: 10,
    padding: '8px 10px',
    background: 'rgba(2,6,23,0.35)',
  },
  summaryLabel: { color: '#94a3b8', fontSize: '0.78rem' },
  summaryValue: { color: '#e2e8f0', fontSize: '0.78rem', fontWeight: 600 },
  rowBetween: { display: 'flex', justifyContent: 'space-between', alignItems: 'center' },
  pre: {
    margin: 0,
    maxHeight: '60vh',
    overflow: 'auto',
    fontSize: '0.78rem',
    lineHeight: 1.45,
    color: '#cbd5e1',
    background: '#020617',
    borderRadius: 10,
    border: '1px solid #1e293b',
    padding: 12,
  },
  subtleLine: {
    margin: '0 0 10px 0',
    color: '#94a3b8',
    fontSize: '0.78rem',
  },
  crownForm: {
    display: 'grid',
    gap: 10,
    marginBottom: 10,
  },
  fileInput: {
    background: '#0f172a',
    color: '#e2e8f0',
    border: '1px solid #334155',
    borderRadius: 10,
    padding: '10px 12px',
    fontFamily: "'JetBrains Mono', monospace",
  },
  errorInline: {
    border: '1px solid rgba(244,63,94,0.45)',
    background: 'rgba(244,63,94,0.12)',
    color: '#fecdd3',
    borderRadius: 10,
    padding: 10,
    fontSize: '0.8rem',
    marginTop: 8,
  },
  tableWrap: {
    overflowX: 'auto',
    border: '1px solid rgba(148,163,184,0.25)',
    borderRadius: 10,
    background: 'rgba(2,6,23,0.45)',
  },
  table: {
    width: '100%',
    borderCollapse: 'collapse',
    fontSize: '0.75rem',
    color: '#e2e8f0',
  },
};
