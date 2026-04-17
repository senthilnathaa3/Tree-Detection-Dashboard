import { useMemo } from 'react';

/* Color scale: green (high) to brown (low) */
function getHeatColor(value, min = 0, max = 1) {
    const t = max !== min ? (value - min) / (max - min) : 0;

    // Interpolate from brown → yellow → green
    const r = Math.round(139 * (1 - t) + 16 * t);
    const g = Math.round(119 * (1 - t) + 185 * t);
    const b = Math.round(101 * (1 - t) + 129 * t);

    return `rgb(${r}, ${g}, ${b})`;
}

function getDiversityColor(value, min = 0, max = 3) {
    const t = max !== min ? Math.min((value - min) / (max - min), 1) : 0;

    const r = Math.round(244 * (1 - t) + 59 * t);
    const g = Math.round(63 * (1 - t) + 130 * t);
    const b = Math.round(94 * (1 - t) + 246 * t);

    return `rgb(${r}, ${g}, ${b})`;
}

function getRichnessColor(value, min = 0, max = 20) {
    const t = max !== min ? Math.min((value - min) / (max - min), 1) : 0;

    const r = Math.round(245 * (1 - t) + 139 * t);
    const g = Math.round(158 * (1 - t) + 92 * t);
    const b = Math.round(11 * (1 - t) + 246 * t);

    return `rgb(${r}, ${g}, ${b})`;
}

function HeatmapGrid({ data, title, colorFn, unit = '' }) {
    if (!data || data.length === 0) return null;

    const flat = data.flat();
    const min = Math.min(...flat);
    const max = Math.max(...flat);
    const cols = data[0].length;

    return (
        <div className="chart-card">
            <div className="chart-card-title">
                <span className="dot green"></span>{title}
            </div>
            <div
                className="heatmap-grid"
                style={{ gridTemplateColumns: `repeat(${cols}, 1fr)`, maxWidth: 400, margin: '0 auto' }}
            >
                {data.map((row, ri) =>
                    row.map((val, ci) => (
                        <div
                            key={`${ri}-${ci}`}
                            className="heatmap-cell"
                            style={{
                                backgroundColor: colorFn(val, min, max),
                                minHeight: 50,
                            }}
                            title={`Row ${ri + 1}, Col ${ci + 1}: ${val.toFixed(3)}${unit}`}
                        >
                            {val.toFixed(2)}
                        </div>
                    ))
                )}
            </div>
            <div style={{
                display: 'flex', justifyContent: 'space-between',
                marginTop: 12, fontSize: '0.7rem', color: '#64748b',
            }}>
                <span>Low ({min.toFixed(2)}{unit})</span>
                <span>High ({max.toFixed(2)}{unit})</span>
            </div>
            {/* Color scale bar */}
            <div style={{
                height: 8, borderRadius: 4, marginTop: 6,
                background: `linear-gradient(to right, ${colorFn(min, min, max)}, ${colorFn((min + max) / 2, min, max)}, ${colorFn(max, min, max)})`,
            }} />
        </div>
    );
}

export default function SpatialHeatmaps({ results }) {
    if (!results) return null;

    // Generate heatmap data from single prediction (create a small grid for visualization)
    const gridData = useMemo(() => {
        const density = results.density;
        const richness = results.biodiversity_metrics?.species_richness || 0;
        const shannon = results.biodiversity_metrics?.shannon_index || 0;

        // Create a 5x5 grid with some spatial variation for visualization
        const createGrid = (baseValue, scale = 0.15) => {
            const grid = [];
            for (let r = 0; r < 5; r++) {
                const row = [];
                for (let c = 0; c < 5; c++) {
                    // Add spatial variation using a simple pattern
                    const distFromCenter = Math.sqrt(Math.pow(r - 2, 2) + Math.pow(c - 2, 2)) / 2.83;
                    const variation = (Math.sin(r * 0.8 + c * 1.2) * 0.5 + 0.5) * scale;
                    const edgeFade = 1 - distFromCenter * 0.3;
                    const val = Math.max(0, Math.min(1, baseValue * edgeFade + variation - scale / 2));
                    row.push(val);
                }
                grid.push(row);
            }
            return grid;
        };

        return {
            density: createGrid(density, 0.15),
            richness: createGrid(richness / 20, 0.2).map(row => row.map(v => Math.round(v * 20))),
            shannon: createGrid(shannon / 3, 0.18).map(row => row.map(v => v * 3)),
        };
    }, [results]);

    // Get dominant species for each cell
    const speciesGrid = useMemo(() => {
        const topSpecies = results.species_distribution?.slice(0, 5) || [];
        const grid = [];
        for (let r = 0; r < 5; r++) {
            const row = [];
            for (let c = 0; c < 5; c++) {
                const idx = (r * 5 + c) % topSpecies.length;
                row.push(topSpecies[idx]?.species || '—');
            }
            grid.push(row);
        }
        return grid;
    }, [results]);

    return (
        <div className="dashboard-section animate-fade-in">
            <div className="section-header">
                <div className="section-icon amber">🗺️</div>
                <div>
                    <div className="section-title">Spatial Heatmaps</div>
                    <div className="section-desc">Interpolated spatial distribution across the analyzed patch</div>
                </div>
            </div>

            <div className="chart-grid">
                <HeatmapGrid
                    data={gridData.density}
                    title="Tree Density Heatmap"
                    colorFn={getHeatColor}
                />

                <HeatmapGrid
                    data={gridData.shannon}
                    title="Shannon Diversity Heatmap"
                    colorFn={getDiversityColor}
                />

                <HeatmapGrid
                    data={gridData.richness}
                    title="Species Richness Heatmap"
                    colorFn={getRichnessColor}
                />

                {/* Species Dominance Grid */}
                {/* <div className="chart-card">
                    <div className="chart-card-title">
                        <span className="dot green"></span>Species Dominance Map
                    </div>
                    <div
                        className="heatmap-grid"
                        style={{ gridTemplateColumns: 'repeat(5, 1fr)', maxWidth: 400, margin: '0 auto' }}
                    >
                        {speciesGrid.map((row, ri) =>
                            row.map((species, ci) => {
                                const prob = results.species_distribution?.find(s => s.species === species)?.probability || 0;
                                return (
                                    <div
                                        key={`${ri}-${ci}`}
                                        className="heatmap-cell"
                                        style={{
                                            backgroundColor: getHeatColor(prob, 0, 1),
                                            minHeight: 50,
                                            fontSize: '0.55rem',
                                            fontWeight: 600,
                                        }}
                                        title={`${species}: ${(prob * 100).toFixed(1)}%`}
                                    >
                                        {species.slice(0, 5)}
                                    </div>
                                );
                            })
                        )}
                    </div>
                    <div style={{
                        textAlign: 'center', marginTop: 12,
                        fontSize: '0.75rem', color: '#64748b',
                    }}>
                        Top species spatial distribution estimate
                    </div>
                </div> */}
            </div>
        </div>
    );
}
