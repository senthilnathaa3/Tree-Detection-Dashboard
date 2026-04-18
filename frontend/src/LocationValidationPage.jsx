import { useEffect, useMemo, useState } from 'react';
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { browsePaths, validateLocationWithCrowns } from './services/api';

const LAST_RUN_STORAGE_KEY = 'location_validation_pitch_v2';

const DEFAULT_FORM = {
  lat: 38.7,
  lon: -79.9,
  radius_km: 10,
  provider: 'planetary_computer',
  start_date: '2024-06-01',
  end_date: '2024-08-31',
  cloud_cover_max: 40,
  threshold: 0.5,
  validation_source: 'fia',
  fia_csv_path: '',
  worldcover_path: '',
  year_start: 2018,
  year_end: 2024,
  calibration_profile_path: '',
  calibration_region: 'WV',
  sample_grid_size: 3,
  crown_radius_km: 0.2,
  crown_ndvi_threshold: 0.6,
  crown_min_area_px: 24,
  crown_max_candidates: 5000,
  crown_align_with_model: true,
  include_pitch_visuals: false,
  representative_imagery_source: 'naip',
};

function fmtNum(v, digits = 1) {
  if (v === null || v === undefined || Number.isNaN(Number(v))) return '—';
  return Number(v).toFixed(digits);
}

function TextField({ label, value, onChange, type = 'text', step, min, max }) {
  return (
    <label style={styles.field}>
      <span style={styles.label}>{label}</span>
      <input
        style={styles.input}
        type={type}
        value={value}
        step={step}
        min={min}
        max={max}
        onChange={(e) => onChange(e.target.value)}
      />
    </label>
  );
}

function SelectField({ label, value, onChange, options }) {
  return (
    <label style={styles.field}>
      <span style={styles.label}>{label}</span>
      <select style={styles.input} value={value} onChange={(e) => onChange(e.target.value)}>
        {options.map((option) => (
          <option key={option.value} value={option.value}>{option.label}</option>
        ))}
      </select>
    </label>
  );
}

function PathField({ label, value, onChange, onBrowse, placeholder }) {
  return (
    <label style={styles.field}>
      <span style={styles.label}>{label}</span>
      <div style={styles.pathRow}>
        <input
          style={styles.input}
          type="text"
          value={value}
          placeholder={placeholder}
          onChange={(e) => onChange(e.target.value)}
        />
        <button type="button" style={styles.ghostBtn} onClick={onBrowse}>Browse</button>
      </div>
    </label>
  );
}

function Kpi({ title, value, subtitle, tone = '#3b82f6' }) {
  return (
    <div style={styles.kpiCard}>
      <div style={{ ...styles.kpiAccent, background: tone }} />
      <div style={styles.kpiTitle}>{title}</div>
      <div style={styles.kpiValue}>{value}</div>
      <div style={styles.kpiSub}>{subtitle}</div>
    </div>
  );
}

function RepresentativeCard({ title, patch }) {
  const crown = patch?.crown_annotation || {};
  const localFia = patch?.fia_local || {};
  const support = patch?.fia_support_strength || 'none';
  const pctDiff = patch?.patch_percent_difference_vs_fia;
  const imagerySource = crown?.imagery_source || 'sentinel';
  return (
    <section style={styles.card}>
      <div style={styles.repHeader}>
        <div>
          <div style={styles.sectionTitle}>{title}</div>
          <div style={styles.smallText}>
            Patch {patch?.patch_id || '—'} · {fmtNum(patch?.calibrated_tph, 1)} TPH calibrated
          </div>
        </div>
        <div style={styles.bucketChip}>{patch?.density_bucket || '—'}</div>
      </div>
      {crown?.annotated_image_data_url ? (
        <img src={crown.annotated_image_data_url} alt={`${title} crown annotation`} style={styles.repImage} />
      ) : (
        <div style={styles.smallText}>
          {crown?.reason || 'Annotated crown image unavailable.'}
        </div>
      )}
      <div style={styles.repCaption}>
        Presentation overlay uses sparse crown markers for readability, not full candidate boxes.
        {crown?.annotated_image_data_url ? ` Visual source: ${imagerySource}.` : ''}
      </div>
      <div style={styles.repFacts}>
        <div><strong>Patch Trees</strong><span>{fmtNum(patch?.patch_tree_count_calibrated, 1)}</span></div>
        <div><strong>Dominant Species</strong><span>{patch?.dominant_species || '—'}</span></div>
        <div><strong>Crown Candidates</strong><span>{fmtNum(crown?.candidate_count, 0)}</span></div>
        <div><strong>Local FIA TPH</strong><span>{fmtNum(localFia?.trees_per_hectare, 1)}</span></div>
        <div><strong>FIA Support</strong><span>{support}</span></div>
        <div><strong>Patch vs FIA</strong><span>{pctDiff === null || pctDiff === undefined ? '—' : `${fmtNum(pctDiff, 1)}%`}</span></div>
      </div>
      <div style={styles.smallText}>{localFia?.note || 'AOI FIA is the primary benchmark for this pitch.'}</div>
    </section>
  );
}

function PathPickerModal({ state, onClose, onOpenPath, onPickPath }) {
  if (!state.open) return null;
  return (
    <div style={styles.modalOverlay} onClick={onClose}>
      <div style={styles.modalCard} onClick={(e) => e.stopPropagation()}>
        <div style={styles.modalHeader}>
          <strong>{state.title || 'Select Path'}</strong>
          <button type="button" style={styles.ghostBtn} onClick={onClose}>Close</button>
        </div>
        <div style={styles.modalPathRow}>
          <button
            type="button"
            style={styles.ghostBtn}
            disabled={!state.parentPath || state.parentPath === state.currentPath}
            onClick={() => onOpenPath(state.parentPath || state.currentPath)}
          >
            Up
          </button>
          <div style={styles.modalPath}>{state.currentPath}</div>
        </div>
        {state.error ? <div style={styles.error}>{state.error}</div> : null}
        <div style={styles.modalList}>
          {state.loading ? (
            <div style={styles.smallText}>Loading...</div>
          ) : (
            state.entries.map((entry) => (
              <button
                key={entry.path}
                type="button"
                style={styles.modalEntry}
                onClick={() => (entry.is_dir ? onOpenPath(entry.path) : onPickPath(entry.path))}
              >
                <span>{entry.is_dir ? '📁' : '📄'}</span>
                <span>{entry.name}</span>
              </button>
            ))
          )}
        </div>
      </div>
    </div>
  );
}

export default function LocationValidationPage() {
  const [form, setForm] = useState(DEFAULT_FORM);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [validation, setValidation] = useState(null);
  const [crowns, setCrowns] = useState(null);
  const [pitchRegions, setPitchRegions] = useState(null);
  const [lastSavedAt, setLastSavedAt] = useState('');
  const [showSidebar, setShowSidebar] = useState(true);

  const [picker, setPicker] = useState({
    open: false,
    title: '',
    targetField: '',
    currentPath: '~',
    parentPath: '',
    entries: [],
    loading: false,
    error: '',
    extensions: [],
  });

  useEffect(() => {
    try {
      const raw = window.localStorage.getItem(LAST_RUN_STORAGE_KEY);
      if (!raw) return;
      const parsed = JSON.parse(raw);
      if (parsed?.form) setForm((prev) => ({ ...prev, ...parsed.form }));
      if (parsed?.validation) setValidation(parsed.validation);
      if (parsed?.crowns) setCrowns(parsed.crowns);
      if (parsed?.pitch_regions) setPitchRegions(parsed.pitch_regions);
      if (parsed?.saved_at) setLastSavedAt(parsed.saved_at);
      setShowSidebar(false);
    } catch {
      // no-op
    }
  }, []);

  const setField = (key, value) => setForm((prev) => ({ ...prev, [key]: value }));

  const openPicker = async (targetField, title, extensions) => {
    const next = {
      open: true,
      title,
      targetField,
      currentPath: '~',
      parentPath: '',
      entries: [],
      loading: true,
      error: '',
      extensions: extensions || [],
    };
    setPicker(next);
    try {
      const data = await browsePaths('~', true, next.extensions);
      setPicker((prev) => ({ ...prev, currentPath: data.current_path, parentPath: data.parent_path, entries: data.entries || [], loading: false }));
    } catch (e) {
      setPicker((prev) => ({ ...prev, loading: false, error: e.message || 'Browse failed' }));
    }
  };

  const browsePickerPath = async (path) => {
    setPicker((prev) => ({ ...prev, loading: true, error: '' }));
    try {
      const data = await browsePaths(path, true, picker.extensions || []);
      setPicker((prev) => ({ ...prev, currentPath: data.current_path, parentPath: data.parent_path, entries: data.entries || [], loading: false }));
    } catch (e) {
      setPicker((prev) => ({ ...prev, loading: false, error: e.message || 'Browse failed' }));
    }
  };

  const submit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError('');

    const payload = {
      lat: Number(form.lat),
      lon: Number(form.lon),
      radius_km: Number(form.radius_km),
      provider: form.provider,
      start_date: form.start_date,
      end_date: form.end_date,
      cloud_cover_max: Number(form.cloud_cover_max),
      threshold: Number(form.threshold),
      validation_source: form.validation_source,
      sample_grid_size: Number(form.sample_grid_size),
      calibration_profile_path: form.calibration_profile_path || undefined,
      calibration_region: form.calibration_region || undefined,
      fia_csv_path: form.validation_source === 'fia' ? form.fia_csv_path : undefined,
      worldcover_path: form.validation_source === 'esa_worldcover' ? form.worldcover_path : undefined,
      year_start: form.year_start === '' ? undefined : Number(form.year_start),
      year_end: form.year_end === '' ? undefined : Number(form.year_end),
      crown_radius_km: Number(form.crown_radius_km),
      crown_ndvi_threshold: Number(form.crown_ndvi_threshold),
      crown_min_area_px: Number(form.crown_min_area_px),
      crown_max_candidates: Number(form.crown_max_candidates),
      crown_align_with_model: Boolean(form.crown_align_with_model),
      include_pitch_visuals: Boolean(form.include_pitch_visuals),
      representative_imagery_source: form.representative_imagery_source,
    };

    try {
      const out = await validateLocationWithCrowns(payload);
      setValidation(out?.validation || null);
      setCrowns(out?.crowns || null);
      setPitchRegions(out?.pitch_regions || null);
      const savedAt = new Date().toISOString();
      setLastSavedAt(savedAt);
      window.localStorage.setItem(
        LAST_RUN_STORAGE_KEY,
        JSON.stringify({
          saved_at: savedAt,
          form,
          validation: out?.validation || null,
          crowns: out?.crowns || null,
          pitch_regions: out?.pitch_regions || null,
        }),
      );
    } catch (err) {
      setError(err.message || 'Run failed');
    } finally {
      setLoading(false);
    }
  };

  const clearCache = () => {
    window.localStorage.removeItem(LAST_RUN_STORAGE_KEY);
    setLastSavedAt('');
  };

  const metrics = useMemo(() => {
    if (!validation) return null;
    const cal = validation?.comparison?.density_agreement_calibrated || null;
    const fia = cal?.fia_mean_trees_per_hectare ?? null;
    const tphCal = cal?.model_tph_calibrated ?? null;
    const pct = cal?.percent_difference ?? null;
    const agreement = pct === null || pct === undefined ? null : Math.max(0, 100 - Math.abs(Number(pct)));
    const crownCount = crowns?.candidate_count ?? null;
    const totalPatches = validation?.model?.summary?.total_tiles_analyzed ?? null;

    const chartData = [
      { metric: 'Model TPH (Cal)', value: tphCal, fill: '#22d3ee' },
      { metric: 'FIA TPH', value: fia, fill: '#34d399' },
    ].filter((x) => x.value !== null && x.value !== undefined);

    const hasCalibration = tphCal !== null && tphCal !== undefined && fia !== null && fia !== undefined;
    const regionChartData = pitchRegions?.summary
      ? [
          { metric: 'High', value: pitchRegions.summary.high?.mean_calibrated_tph, fill: '#f97316' },
          { metric: 'Medium', value: pitchRegions.summary.medium?.mean_calibrated_tph, fill: '#facc15' },
          { metric: 'Low', value: pitchRegions.summary.low?.mean_calibrated_tph, fill: '#22c55e' },
        ].filter((x) => x.value !== null && x.value !== undefined)
      : [];
    return { tphCal, fia, pct, agreement, crownCount, totalPatches, chartData, hasCalibration, regionChartData };
  }, [validation, crowns, pitchRegions]);

  const representativeByBucket = useMemo(() => {
    const reps = pitchRegions?.representatives || [];
    return {
      high: reps.find((x) => x.density_bucket === 'high') || null,
      medium: reps.find((x) => x.density_bucket === 'medium') || null,
      low: reps.find((x) => x.density_bucket === 'low') || null,
    };
  }, [pitchRegions]);

  return (
    <div style={styles.page}>
      <header style={{ ...styles.header, maxWidth: showSidebar ? 1280 : '100%' }}>
        <div>
          <h1 style={styles.h1}>Location Validation Pitch</h1>
          <p style={styles.sub}>Calibrated TPH + Crown Detection in one run</p>
        </div>
        <div style={styles.headerActions}>
          <button type="button" style={styles.ghostBtn} onClick={() => setShowSidebar((v) => !v)}>
            {showSidebar ? 'Hide Configuration' : 'Edit Configuration'}
          </button>
          <a href="/" style={styles.ghostBtn}>Open Dashboard</a>
        </div>
      </header>

      {lastSavedAt ? (
        <div style={{ ...styles.cacheBanner, maxWidth: showSidebar ? 1280 : '100%' }}>
          <span>Loaded cached run from {new Date(lastSavedAt).toLocaleString()}</span>
          <button type="button" style={styles.ghostBtn} onClick={clearCache}>Clear Cache</button>
        </div>
      ) : null}

      <div
        style={{
          ...styles.layout,
          maxWidth: showSidebar ? 1280 : '100%',
          gridTemplateColumns: showSidebar ? '360px minmax(0, 1fr)' : 'minmax(0, 1fr)',
        }}
      >
        {showSidebar ? (
          <aside style={styles.sidebar}>
          <form onSubmit={submit} style={styles.card}>
            <div style={styles.sectionTitle}>Validation Inputs</div>
            <TextField label="Latitude" value={form.lat} onChange={(v) => setField('lat', v)} type="number" step="0.0001" min={-90} max={90} />
            <TextField label="Longitude" value={form.lon} onChange={(v) => setField('lon', v)} type="number" step="0.0001" min={-180} max={180} />
            <TextField label="AOI Radius (km)" value={form.radius_km} onChange={(v) => setField('radius_km', v)} type="number" step="0.1" min={0.1} />
            <TextField label="Start Date" value={form.start_date} onChange={(v) => setField('start_date', v)} type="date" />
            <TextField label="End Date" value={form.end_date} onChange={(v) => setField('end_date', v)} type="date" />
            <TextField label="Cloud Cover Max" value={form.cloud_cover_max} onChange={(v) => setField('cloud_cover_max', v)} type="number" step="1" min={0} max={100} />
            <TextField label="Sample Grid" value={form.sample_grid_size} onChange={(v) => setField('sample_grid_size', v)} type="number" step="1" min={1} />
            <PathField
              label="Calibration Profile (.json)"
              value={form.calibration_profile_path}
              onChange={(v) => setField('calibration_profile_path', v)}
              placeholder="/abs/path/regional_calibration.json"
              onBrowse={() => openPicker('calibration_profile_path', 'Select Calibration Profile', ['.json'])}
            />
            <TextField label="Calibration Region" value={form.calibration_region} onChange={(v) => setField('calibration_region', v)} />
            <PathField
              label="FIA CSV"
              value={form.fia_csv_path}
              onChange={(v) => setField('fia_csv_path', v)}
              placeholder="/abs/path/fia_multi_state.csv"
              onBrowse={() => openPicker('fia_csv_path', 'Select FIA CSV', ['.csv'])}
            />

            <div style={styles.sectionTitle}>Crown Inputs</div>
            <TextField label="Crown Radius (km)" value={form.crown_radius_km} onChange={(v) => setField('crown_radius_km', v)} type="number" step="0.05" min={0.05} />
            <TextField label="NDVI Threshold" value={form.crown_ndvi_threshold} onChange={(v) => setField('crown_ndvi_threshold', v)} type="number" step="0.01" min={-1} max={1} />
            <TextField label="Min Crown Area (px)" value={form.crown_min_area_px} onChange={(v) => setField('crown_min_area_px', v)} type="number" step="1" min={1} />
            <TextField label="Max Candidates" value={form.crown_max_candidates} onChange={(v) => setField('crown_max_candidates', v)} type="number" step="1" min={1} />
            <SelectField
              label="Representative Imagery"
              value={form.representative_imagery_source}
              onChange={(v) => setField('representative_imagery_source', v)}
              options={[
                { value: 'naip', label: 'NAIP Only' },
                { value: 'auto', label: 'Auto Fallback' },
                { value: 'sentinel', label: 'Sentinel Only' },
              ]}
            />

            <label style={{ ...styles.field, display: 'flex', gap: 8, alignItems: 'center', marginTop: 6 }}>
              <input
                type="checkbox"
                checked={Boolean(form.crown_align_with_model)}
                onChange={(e) => setField('crown_align_with_model', e.target.checked)}
              />
              <span style={{ fontSize: '0.82rem', color: 'var(--text-secondary)' }}>Align crown count with model TPH</span>
            </label>

            <label style={{ ...styles.field, display: 'flex', gap: 8, alignItems: 'center', marginTop: 2 }}>
              <input
                type="checkbox"
                checked={Boolean(form.include_pitch_visuals)}
                onChange={(e) => setField('include_pitch_visuals', e.target.checked)}
              />
              <span style={{ fontSize: '0.82rem', color: 'var(--text-secondary)' }}>
                Generate representative patch visuals
              </span>
            </label>
            <div style={styles.smallText}>
              Off: faster AOI density + FIA zoning. On: slower run with annotated patch cards.
            </div>

            <button type="submit" style={styles.primaryBtn} disabled={loading}>
              {loading ? 'Running…' : form.include_pitch_visuals ? 'Run Full Pitch' : 'Run Fast Density'}
            </button>
          </form>
          </aside>
        ) : null}

        <main style={styles.main}>
          {error ? <div style={styles.error}>{error}</div> : null}

          {metrics?.hasCalibration ? (
            <>
              <section style={styles.card}>
                <div style={styles.sectionTitle}>Pitch KPIs (Calibrated Only)</div>
                <div style={styles.kpiGrid}>
                  <Kpi title="Agreement Score" value={metrics.agreement === null ? '—' : `${fmtNum(metrics.agreement, 2)}%`} subtitle="100 - |calibrated % diff|" tone="#a78bfa" />
                  <Kpi title="Model TPH (Calibrated)" value={fmtNum(metrics.tphCal, 2)} subtitle="Trees/hectare" tone="#22d3ee" />
                  <Kpi title="FIA TPH" value={fmtNum(metrics.fia, 2)} subtitle="AOI reference" tone="#34d399" />
                  <Kpi title="AOI Patches" value={fmtNum(metrics.totalPatches, 0)} subtitle="Grid samples analyzed" tone="#f97316" />
                </div>
              </section>

              <section style={styles.card}>
                <div style={styles.sectionTitle}>AOI TPH vs FIA</div>
                <ResponsiveContainer width="100%" height={280}>
                  <BarChart data={metrics.chartData} margin={{ top: 10, right: 16, left: 0, bottom: 8 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.25)" />
                    <XAxis dataKey="metric" tick={{ fill: 'var(--text-muted)', fontSize: 12 }} />
                    <YAxis tick={{ fill: 'var(--text-muted)', fontSize: 12 }} />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: 'var(--bg-elevated)',
                        border: '1px solid var(--border-color)',
                        boxShadow: 'none',
                      }}
                    />
                    <Bar dataKey="value" radius={[8, 8, 0, 0]}>
                      {metrics.chartData.map((e) => <Cell key={e.metric} fill={e.fill} />)}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </section>

              {pitchRegions?.status === 'success' ? (
                <section style={styles.card}>
                  <div style={styles.sectionTitle}>Density Regions</div>
                  <div style={styles.kpiGrid}>
                    <Kpi title="High Density" value={fmtNum(pitchRegions.summary?.high?.mean_calibrated_tph, 1)} subtitle={`${fmtNum(pitchRegions.summary?.high?.count, 0)} patches`} tone="#f97316" />
                    <Kpi title="Medium Density" value={fmtNum(pitchRegions.summary?.medium?.mean_calibrated_tph, 1)} subtitle={`${fmtNum(pitchRegions.summary?.medium?.count, 0)} patches`} tone="#facc15" />
                    <Kpi title="Low Density" value={fmtNum(pitchRegions.summary?.low?.mean_calibrated_tph, 1)} subtitle={`${fmtNum(pitchRegions.summary?.low?.count, 0)} patches`} tone="#22c55e" />
                    <Kpi title="Crown Candidates" value={fmtNum(metrics.crownCount, 0)} subtitle="Overview patch crowns" tone="#38bdf8" />
                  </div>
                </section>
              ) : null}

              {metrics.regionChartData?.length ? (
                <section style={styles.card}>
                  <div style={styles.sectionTitle}>Region Mean TPH</div>
                  <ResponsiveContainer width="100%" height={260}>
                    <BarChart data={metrics.regionChartData} margin={{ top: 10, right: 16, left: 0, bottom: 8 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.25)" />
                      <XAxis dataKey="metric" tick={{ fill: 'var(--text-muted)', fontSize: 12 }} />
                      <YAxis tick={{ fill: 'var(--text-muted)', fontSize: 12 }} />
                      <Tooltip contentStyle={{ backgroundColor: 'var(--bg-elevated)', border: '1px solid var(--border-color)', boxShadow: 'none' }} />
                      <Bar dataKey="value" radius={[8, 8, 0, 0]}>
                        {metrics.regionChartData.map((e) => <Cell key={e.metric} fill={e.fill} />)}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </section>
              ) : null}

              {(representativeByBucket.high || representativeByBucket.medium || representativeByBucket.low) ? (
                <section style={styles.repGrid}>
                  {representativeByBucket.high ? <RepresentativeCard title="High Density Patch" patch={representativeByBucket.high} /> : null}
                  {representativeByBucket.medium ? <RepresentativeCard title="Medium Density Patch" patch={representativeByBucket.medium} /> : null}
                  {representativeByBucket.low ? <RepresentativeCard title="Low Density Patch" patch={representativeByBucket.low} /> : null}
                </section>
              ) : null}
            </>
          ) : (
            <section style={styles.card}>
              <div style={styles.smallText}>
                Calibrated metrics are hidden until calibration is available.
                Provide `calibration_profile_path` and `calibration_region`, then run again.
              </div>
            </section>
          )}
        </main>
      </div>

      <PathPickerModal
        state={picker}
        onClose={() => setPicker((p) => ({ ...p, open: false }))}
        onOpenPath={browsePickerPath}
        onPickPath={(path) => {
          if (picker.targetField) setField(picker.targetField, path);
          setPicker((p) => ({ ...p, open: false }));
        }}
      />
    </div>
  );
}

const styles = {
  page: {
    minHeight: '100vh',
    background: 'var(--bg-primary)',
    color: 'var(--text-primary)',
    padding: '24px',
    fontFamily: 'var(--font-sans)',
  },
  header: {
    maxWidth: 1280,
    margin: '0 auto 14px',
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    gap: 12,
  },
  h1: { margin: 0, fontSize: '1.45rem' },
  sub: { margin: '6px 0 0', color: 'var(--text-muted)', fontSize: '0.88rem' },
  cacheBanner: {
    maxWidth: 1280,
    margin: '0 auto 12px',
    border: '1px solid rgba(14,165,233,0.35)',
    background: 'rgba(14,165,233,0.08)',
    borderRadius: 10,
    padding: '8px 10px',
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    gap: 10,
    fontSize: '0.78rem',
  },
  layout: {
    maxWidth: 1280,
    margin: '0 auto',
    display: 'grid',
    gridTemplateColumns: '360px 1fr',
    gap: 18,
    alignItems: 'start',
  },
  sidebar: { position: 'sticky', top: 12, maxHeight: 'calc(100vh - 24px)', overflowY: 'auto' },
  main: { minWidth: 0, display: 'grid', gap: 14 },
  card: {
    background: 'var(--bg-card)',
    border: '1px solid var(--border-color)',
    borderRadius: 12,
    padding: 14,
    display: 'grid',
    gap: 10,
  },
  sectionTitle: { fontSize: '0.95rem', fontWeight: 700, marginBottom: 4 },
  field: { display: 'grid', gap: 6 },
  label: { color: 'var(--text-secondary)', fontSize: '0.72rem', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.05em' },
  input: {
    background: 'var(--bg-input)',
    border: '1px solid var(--border-color)',
    color: 'var(--text-primary)',
    borderRadius: 8,
    padding: '9px 10px',
    fontSize: '0.84rem',
    outline: 'none',
  },
  pathRow: { display: 'grid', gridTemplateColumns: '1fr auto', gap: 8, alignItems: 'center' },
  primaryBtn: {
    background: 'var(--green-600)',
    color: 'white',
    border: 'none',
    borderRadius: 8,
    padding: '10px 12px',
    fontWeight: 700,
    cursor: 'pointer',
    marginTop: 8,
  },
  ghostBtn: {
    background: 'transparent',
    color: 'var(--text-secondary)',
    border: '1px solid var(--border-color)',
    borderRadius: 8,
    padding: '7px 10px',
    textDecoration: 'none',
    cursor: 'pointer',
    fontSize: '0.78rem',
  },
  error: {
    border: '1px solid rgba(244,63,94,0.35)',
    background: 'rgba(244,63,94,0.08)',
    color: '#fecdd3',
    borderRadius: 8,
    padding: '8px 10px',
    fontSize: '0.8rem',
  },
  kpiGrid: { display: 'grid', gridTemplateColumns: 'repeat(auto-fit,minmax(210px,1fr))', gap: 12 },
  kpiCard: {
    position: 'relative',
    background: 'var(--bg-elevated)',
    border: '1px solid var(--border-color)',
    borderRadius: 10,
    padding: '12px 12px 12px 16px',
  },
  kpiAccent: { position: 'absolute', left: 0, top: 0, bottom: 0, width: 4, borderRadius: '10px 0 0 10px' },
  kpiTitle: { color: 'var(--text-muted)', fontSize: '0.72rem', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.05em' },
  kpiValue: { fontSize: '1.5rem', fontWeight: 700, marginTop: 4 },
  kpiSub: { color: 'var(--text-muted)', fontSize: '0.72rem', marginTop: 2 },
  tableWrap: { overflowX: 'auto', border: '1px solid var(--border-color)', borderRadius: 10 },
  table: { width: '100%', borderCollapse: 'collapse', fontSize: '0.8rem' },
  smallText: { color: 'var(--text-muted)', fontSize: '0.8rem' },
  repGrid: { display: 'grid', gridTemplateColumns: 'repeat(auto-fit,minmax(300px,1fr))', gap: 14 },
  repHeader: { display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: 10 },
  bucketChip: {
    border: '1px solid var(--border-color)',
    background: 'var(--bg-elevated)',
    borderRadius: 999,
    padding: '4px 9px',
    textTransform: 'uppercase',
    fontSize: '0.68rem',
    color: 'var(--text-secondary)',
    letterSpacing: '0.05em',
  },
  repImage: {
    width: '100%',
    aspectRatio: '1 / 1',
    objectFit: 'cover',
    borderRadius: 10,
    border: '1px solid var(--border-color)',
    background: 'var(--bg-elevated)',
  },
  repCaption: {
    color: 'var(--text-muted)',
    fontSize: '0.74rem',
    lineHeight: 1.4,
  },
  repFacts: {
    display: 'grid',
    gridTemplateColumns: '1fr 1fr',
    gap: 8,
  },
  modalOverlay: {
    position: 'fixed',
    inset: 0,
    background: 'rgba(2,6,23,0.78)',
    zIndex: 1200,
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    padding: 16,
  },
  modalCard: {
    width: 'min(900px, 96vw)',
    maxHeight: '88vh',
    background: 'var(--bg-card)',
    border: '1px solid var(--border-color)',
    borderRadius: 12,
    padding: 12,
    display: 'grid',
    gridTemplateRows: 'auto auto 1fr',
    gap: 10,
  },
  modalHeader: { display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 10 },
  modalPathRow: { display: 'grid', gridTemplateColumns: 'auto 1fr', gap: 8, alignItems: 'center' },
  modalPath: {
    border: '1px solid var(--border-color)',
    borderRadius: 8,
    padding: '8px 10px',
    color: 'var(--text-secondary)',
    fontSize: '0.78rem',
    overflow: 'hidden',
    textOverflow: 'ellipsis',
    whiteSpace: 'nowrap',
  },
  modalList: { overflowY: 'auto', border: '1px solid var(--border-color)', borderRadius: 8, padding: 8, display: 'grid', gap: 6 },
  modalEntry: {
    background: 'var(--bg-elevated)',
    color: 'var(--text-primary)',
    border: '1px solid var(--border-color)',
    borderRadius: 8,
    padding: '8px 10px',
    display: 'grid',
    gridTemplateColumns: '20px 1fr',
    gap: 8,
    textAlign: 'left',
    cursor: 'pointer',
    fontSize: '0.82rem',
  },
};
