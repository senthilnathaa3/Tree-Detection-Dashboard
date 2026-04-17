import { useMemo, useState } from 'react';
import { validateLocation } from './services/api';

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

export default function LocationValidationPage() {
  const [form, setForm] = useState(FOREST_PRESET);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [result, setResult] = useState(null);

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

  return (
    <div style={styles.page}>
      <header style={styles.header}>
        <div>
          <h1 style={styles.h1}>Lat/Lon Validation</h1>
          <p style={styles.sub}>Run remote Sentinel inference and validate with FIA or ESA WorldCover</p>
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

      {result ? (
        <section style={styles.resultCard}>
          <h2 style={styles.h2}>Result</h2>
          <pre style={styles.pre}>{JSON.stringify(result, null, 2)}</pre>
        </section>
      ) : null}
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
    maxWidth: 1200,
    margin: '0 auto 16px',
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
    gap: 16,
  },
  h1: { margin: 0, fontSize: '1.4rem' },
  h2: { margin: '0 0 12px 0', fontSize: '1.1rem' },
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
    maxWidth: 1200,
    margin: '0 auto',
    background: 'rgba(15, 23, 42, 0.7)',
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
    maxWidth: 1200,
    margin: '14px auto 0',
    borderRadius: 10,
    border: '1px solid rgba(244,63,94,0.45)',
    background: 'rgba(244,63,94,0.12)',
    color: '#fecdd3',
    padding: 12,
    fontSize: '0.85rem',
  },
  resultCard: {
    maxWidth: 1200,
    margin: '14px auto 0',
    background: 'rgba(15, 23, 42, 0.7)',
    border: '1px solid rgba(148, 163, 184, 0.2)',
    borderRadius: 14,
    padding: 16,
  },
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
};
