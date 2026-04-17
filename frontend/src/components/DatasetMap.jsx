import { useEffect, useState } from 'react';
import { MapContainer, TileLayer, Rectangle, useMap } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import { getDatasetBounds } from '../services/api';

// Auto-fit the map to the bounds when they change
function FitBounds({ bounds }) {
    const map = useMap();
    useEffect(() => {
        if (bounds) {
            map.fitBounds(bounds, { padding: [40, 40], maxZoom: 15 });
        }
    }, [bounds, map]);
    return null;
}

/**
 * DatasetMap supports two modes:
 * 1. datasetPath  → fetches bounds from /dataset-bounds endpoint (folder mode)
 * 2. boundsData   → uses pre-computed bounds directly (single file mode)
 *
 * Props: { datasetPath?, boundsData?, label? }
 */
export default function DatasetMap({ datasetPath, boundsData: propBounds, label }) {
    const [boundsData, setBoundsData] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    useEffect(() => {
        // If direct bounds are provided (single-file mode), use them directly
        if (propBounds) {
            setBoundsData(propBounds);
            setLoading(false);
            setError(null);
            return;
        }

        // If a dataset path is provided (folder mode), fetch from API
        if (!datasetPath) {
            setBoundsData(null);
            return;
        }

        let cancelled = false;
        const fetchBounds = async () => {
            setLoading(true);
            setError(null);
            try {
                const data = await getDatasetBounds(datasetPath);
                if (!cancelled) setBoundsData(data);
            } catch (err) {
                if (!cancelled) setError(err.message);
            } finally {
                if (!cancelled) setLoading(false);
            }
        };

        fetchBounds();
        return () => { cancelled = true; };
    }, [datasetPath, propBounds]);

    // Loading state
    if (loading) {
        return (
            <div className="dashboard-section animate-fade-in">
                <div className="section-header">
                    <div className="section-icon teal">🛰️</div>
                    <div>
                        <div className="section-title">Dataset Geographic Coverage</div>
                        <div className="section-desc">Loading geographic bounds...</div>
                    </div>
                </div>
                <div className="chart-card" style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', minHeight: 200 }}>
                    <div className="spinner" style={{ width: 28, height: 28, borderWidth: 3 }}></div>
                    <span style={{ marginLeft: 12, color: '#94a3b8', fontSize: '0.85rem' }}>Scanning GeoTIFF metadata...</span>
                </div>
            </div>
        );
    }

    // Error state
    if (error) {
        return (
            <div className="dashboard-section animate-fade-in">
                <div className="section-header">
                    <div className="section-icon teal">🛰️</div>
                    <div>
                        <div className="section-title">Dataset Geographic Coverage</div>
                        <div className="section-desc">Could not extract geographic bounds</div>
                    </div>
                </div>
                <div className="chart-card" style={{
                    display: 'flex', alignItems: 'center', justifyContent: 'center',
                    minHeight: 120, color: '#f43f5e', fontSize: '0.85rem'
                }}>
                    ⚠️ {error}
                </div>
            </div>
        );
    }

    // No data yet
    if (!boundsData) return null;

    const { west, south, east, north, center_lat, center_lon, area_km2, total_tiles } = boundsData;
    const rectangleBounds = [[south, west], [north, east]];

    const subtitle = label || (
        total_tiles === 1
            ? 'Geographic location of the uploaded Sentinel image'
            : 'Bounding area of uploaded Sentinel imagery'
    );

    return (
        <div className="dashboard-section animate-fade-in">
            {/* Section Header */}
            <div className="section-header">
                <div className="section-icon teal">🛰️</div>
                <div>
                    <div className="section-title">
                        {total_tiles === 1 ? 'Image Geographic Location' : 'Dataset Geographic Coverage'}
                    </div>
                    <div className="section-desc">{subtitle}</div>
                </div>
            </div>

            <div className="chart-card" style={{ padding: 0, overflow: 'hidden' }}>
                {/* Map */}
                <div style={{ height: 400, position: 'relative' }}>
                    <MapContainer
                        center={[center_lat, center_lon]}
                        zoom={13}
                        style={{ height: '100%', width: '100%', borderRadius: 'var(--radius-lg) var(--radius-lg) 0 0' }}
                        scrollWheelZoom={true}
                        zoomControl={true}
                    >
                        <TileLayer
                            attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
                            url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                        />
                        <Rectangle
                            bounds={rectangleBounds}
                            pathOptions={{
                                color: '#10b981',
                                weight: 2,
                                fillColor: '#10b981',
                                fillOpacity: 0.15,
                                dashArray: '6 4',
                            }}
                        />
                        <FitBounds bounds={rectangleBounds} />
                    </MapContainer>
                </div>

                {/* Stats Bar */}
                <div style={{
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-around',
                    padding: '14px 20px',
                    borderTop: '1px solid var(--border-color)',
                    background: 'var(--bg-card)',
                    flexWrap: 'wrap',
                    gap: 12,
                }}>
                    <MapStat icon="📐" label="Area" value={`${area_km2} km²`} />
                    <MapStat icon="📍" label="Center" value={`${center_lat.toFixed(4)}, ${center_lon.toFixed(4)}`} />
                    <MapStat icon="🗂️" label="Total Tiles" value={total_tiles.toLocaleString()} />
                    <MapStat
                        icon="🧭"
                        label="Bounds"
                        value={`${south.toFixed(3)}°–${north.toFixed(3)}°N, ${west.toFixed(3)}°–${east.toFixed(3)}°E`}
                    />
                </div>
            </div>
        </div>
    );
}

function MapStat({ icon, label, value }) {
    return (
        <div style={{ textAlign: 'center', minWidth: 90 }}>
            <div style={{
                fontSize: '0.68rem', color: '#64748b',
                textTransform: 'uppercase', letterSpacing: '0.05em', fontWeight: 600,
            }}>
                {icon} {label}
            </div>
            <div style={{
                fontSize: '0.82rem', fontWeight: 700, color: '#f1f5f9',
                fontFamily: "'JetBrains Mono', monospace", marginTop: 2,
            }}>
                {value}
            </div>
        </div>
    );
}
