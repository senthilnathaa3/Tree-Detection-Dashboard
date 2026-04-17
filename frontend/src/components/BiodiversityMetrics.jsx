import {
    ResponsiveContainer,
    ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, Cell,
    BarChart, Bar,
} from 'recharts';

const tooltipStyle = {
    backgroundColor: '#1a2332',
    border: '1px solid rgba(148, 163, 184, 0.2)',
    borderRadius: '10px',
    color: '#f1f5f9',
    fontSize: '0.8rem',
};

const tooltipLabelStyle = { color: '#f1f5f9' };
const tooltipItemStyle = { color: '#e2e8f0' };

/* ---- Shannon Gauge ---- */
function ShannonGauge({ value, maxValue = 3.0 }) {
    const pct = Math.min(value / maxValue, 1);
    const barWidth = 280;
    const filled = pct * barWidth;

    let color = '#f43f5e';
    if (pct > 0.6) color = '#10b981';
    else if (pct > 0.3) color = '#f59e0b';

    return (
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', padding: '20px 0' }}>
            <svg width={barWidth + 20} height={60} viewBox={`0 0 ${barWidth + 20} 60`}>
                {/* Background */}
                <rect x={10} y={15} width={barWidth} height={24} rx={12} fill="#243042" />
                {/* Fill */} 
                <rect
                    x={10} y={15}
                    width={filled} height={24}
                    rx={12}
                    fill={color}
                    style={{ transition: 'width 1.2s cubic-bezier(0.4, 0, 0.2, 1)' }}
                />
                {/* Labels */}
                <text x={10} y={55} fill="#64748b" fontSize={10}>0</text>
                <text x={barWidth / 2 + 10} y={55} fill="#64748b" fontSize={10} textAnchor="middle">{(maxValue / 2).toFixed(1)}</text>
                <text x={barWidth + 10} y={55} fill="#64748b" fontSize={10} textAnchor="end">{maxValue.toFixed(1)}</text>
                {/* Value indicator */}
                <text x={filled + 10} y={10} fill={color} fontSize={13} fontWeight={700} textAnchor="middle">
                    {value.toFixed(3)}
                </text>
            </svg>
            <span style={{ fontSize: '0.8rem', color: '#94a3b8', marginTop: 8 }}>Shannon Diversity Index</span>
        </div>
    );
}

/* ---- Biodiversity Score Card ---- */
function BioScoreCard({ label, value, unit, icon, color }) {
    return (
        <div style={{
            background: 'rgba(26, 35, 50, 0.8)',
            border: '1px solid rgba(148, 163, 184, 0.1)',
            borderRadius: 12, padding: '16px 20px',
            display: 'flex', alignItems: 'center', gap: 16,
            flex: '1 1 200px',
        }}>
            <div style={{
                width: 44, height: 44, borderRadius: 10,
                background: `${color}20`,
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                fontSize: '1.3rem',
            }}>
                {icon}
            </div>
            <div>
                <div style={{ fontSize: '0.7rem', color: '#64748b', textTransform: 'uppercase', letterSpacing: '0.1em', fontWeight: 600 }}>
                    {label}
                </div>
                <div style={{ fontSize: '1.4rem', fontWeight: 800, color }}>
                    {typeof value === 'number' ? value.toFixed(4) : value}
                    {unit && <span style={{ fontSize: '0.8rem', fontWeight: 400, color: '#94a3b8' }}> {unit}</span>}
                </div>
            </div>
        </div>
    );
}

export default function BiodiversityMetrics({ results }) {
    if (!results) return null;

    const bio = results.biodiversity_metrics;
    if (!bio) return null;

    // Data for Richness vs Evenness scatter
    const scatterData = [
        {
            richness: bio.species_richness,
            evenness: bio.evenness,
            label: 'Current Patch',
        },
        // Add reference points
        { richness: 1, evenness: 0, label: 'Monoculture' },
        { richness: 20, evenness: 1, label: 'Max Diversity' },
        { richness: 10, evenness: 0.5, label: 'Moderate' },
    ];

    const scatterColors = ['#10b981', '#f43f5e', '#3b82f6', '#f59e0b'];

    // Metrics breakdown bar
    const metricsBar = [
        { name: 'Shannon', value: bio.shannon_index, max: 3 },
        { name: 'Evenness', value: bio.evenness, max: 1 },
        { name: 'Simpson', value: bio.simpsons_index, max: 1 },
        { name: 'Bio Score', value: bio.biodiversity_score / 100, max: 1 },
    ];

    return (
        <div className="dashboard-section animate-fade-in">
            <div className="section-header">
                <div className="section-icon purple">🧬</div>
                <div>
                    <div className="section-title">Biodiversity Metrics</div>
                    <div className="section-desc">Ecological diversity indices and analysis</div>
                </div>
            </div>

            {/* <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap', marginBottom: 24 }}>
                <BioScoreCard
                    label="Species Richness"
                    value={bio.species_richness}
                    unit="species"
                    icon="🌱"
                    color="#10b981"
                />
                <BioScoreCard
                    label="Shannon Index"
                    value={bio.shannon_index}
                    icon="📊"
                    color="#3b82f6"
                />
                <BioScoreCard
                    label="Evenness"
                    value={bio.evenness}
                    icon="⚖️"
                    color="#f59e0b"
                />
                <BioScoreCard
                    label="Simpson's Index"
                    value={bio.simpsons_index}
                    icon="🔬"
                    color="#a78bfa"
                />
            </div> */}

            <div className="chart-grid">
                {/* <div className="chart-card">
                    <div className="chart-card-title">
                        <span className="dot blue"></span>Shannon Diversity Gauge
                    </div>
                    <ShannonGauge value={bio.shannon_index} />

                    <div style={{ marginTop: 16 }}>
                        {metricsBar.map((m) => (
                            <div key={m.name} style={{ marginBottom: 12 }}>
                                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', color: '#94a3b8', marginBottom: 4 }}>
                                    <span>{m.name}</span>
                                    <span style={{ fontFamily: "'JetBrains Mono', monospace", color: '#f1f5f9' }}>
                                        {m.value.toFixed(4)}
                                    </span>
                                </div>
                                <div className="progress-bar">
                                    <div
                                        className="progress-fill"
                                        style={{ width: `${(m.value / m.max) * 100}%` }}
                                    />
                                </div>
                            </div>
                        ))}
                    </div>
                </div> */}

                {/* <div className="chart-card">
                    <div className="chart-card-title">
                        <span className="dot green"></span>Richness vs Evenness
                    </div>
                    <ResponsiveContainer width="100%" height={320}>
                        <ScatterChart margin={{ top: 10, right: 20, bottom: 10, left: 10 }}>
                            <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.1)" />
                            <XAxis
                                type="number"
                                dataKey="richness"
                                name="Richness"
                                domain={[0, 22]}
                                tick={{ fill: '#94a3b8', fontSize: 12 }}
                                axisLine={{ stroke: 'rgba(148,163,184,0.1)' }}
                                label={{ value: 'Species Richness', position: 'bottom', fill: '#64748b', fontSize: 11 }}
                            />
                            <YAxis
                                type="number"
                                dataKey="evenness"
                                name="Evenness"
                                domain={[0, 1.1]}
                                tick={{ fill: '#94a3b8', fontSize: 12 }}
                                axisLine={{ stroke: 'rgba(148,163,184,0.1)' }}
                                label={{ value: 'Evenness', angle: -90, position: 'insideLeft', fill: '#64748b', fontSize: 11 }}
                            />
                            <Tooltip
                                contentStyle={tooltipStyle}
                                formatter={(value, name) => [value.toFixed(4), name]}
                                labelFormatter={(_, payload) => payload?.[0]?.payload?.label || ''}
                            />
                            <Scatter data={scatterData} animationDuration={1200}>
                                {scatterData.map((_, i) => (
                                    <Cell key={i} fill={scatterColors[i]} r={i === 0 ? 10 : 6} />
                                ))}
                            </Scatter>
                        </ScatterChart>
                    </ResponsiveContainer>
                    <div style={{ display: 'flex', gap: 16, justifyContent: 'center', marginTop: 8, fontSize: '0.7rem' }}>
                        {scatterData.map((d, i) => (
                            <span key={d.label} style={{ display: 'flex', alignItems: 'center', gap: 4, color: '#94a3b8' }}>
                                <span style={{ width: 8, height: 8, borderRadius: '50%', background: scatterColors[i], display: 'inline-block' }}></span>
                                {d.label}
                            </span>
                        ))}
                    </div>
                </div> */}

                {/* Biodiversity Overview */}
                <div className="chart-card full-width">
                    <div className="chart-card-title">
                        <span className="dot purple"></span>Biodiversity Score Overview
                    </div>
                    <div style={{
                        display: 'flex', alignItems: 'center', justifyContent: 'center',
                        padding: '30px 0', gap: 40,
                    }}>
                        <div style={{ position: 'relative', width: 160, height: 160 }}>
                            <svg viewBox="0 0 160 160" width={160} height={160}>
                                <circle cx={80} cy={80} r={70} fill="none" stroke="#243042" strokeWidth={10} />
                                <circle
                                    cx={80} cy={80} r={70}
                                    fill="none"
                                    stroke="#10b981"
                                    strokeWidth={10}
                                    strokeLinecap="round"
                                    strokeDasharray={`${(bio.biodiversity_score / 100) * 440} 440`}
                                    transform="rotate(-90 80 80)"
                                    style={{ transition: 'stroke-dasharray 1.5s cubic-bezier(0.4, 0, 0.2, 1)' }}
                                />
                            </svg>
                            <div style={{
                                position: 'absolute', inset: 0,
                                display: 'flex', flexDirection: 'column',
                                alignItems: 'center', justifyContent: 'center',
                            }}>
                                <div style={{ fontSize: '2rem', fontWeight: 800, color: '#10b981' }}>
                                    {bio.biodiversity_score.toFixed(1)}
                                </div>
                                <div style={{ fontSize: '0.7rem', color: '#64748b' }}>/ 100</div>
                            </div>
                        </div>

                        <div style={{ maxWidth: 300 }}>
                            <h4 style={{ color: '#f1f5f9', marginBottom: 8 }}>
                                {bio.biodiversity_score >= 70 ? '🌿 High Biodiversity' :
                                    bio.biodiversity_score >= 40 ? '🌱 Moderate Biodiversity' :
                                        '⚠️ Low Biodiversity'}
                            </h4>
                            <p style={{ fontSize: '0.85rem', color: '#94a3b8', lineHeight: 1.6 }}>
                                {bio.biodiversity_score >= 70
                                    ? 'This patch shows excellent species diversity with well-distributed populations across multiple tree species.'
                                    : bio.biodiversity_score >= 40
                                        ? 'Moderate species diversity detected. Some species dominate while others have lower representation.'
                                        : 'Limited species diversity detected. The patch may be dominated by one or few species.'}
                            </p>
                            <div style={{ marginTop: 12, display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                                {/* <span style={{
                                    padding: '3px 10px', borderRadius: 20, fontSize: '0.7rem', fontWeight: 600,
                                    background: 'rgba(16, 185, 129, 0.15)', color: '#10b981',
                                }}>
                                    {bio.species_richness} Species
                                </span> */}
                                {/* <span style={{
                                    padding: '3px 10px', borderRadius: 20, fontSize: '0.7rem', fontWeight: 600,
                                    background: 'rgba(59, 130, 246, 0.15)', color: '#3b82f6',
                                }}>
                                    H = {bio.shannon_index.toFixed(3)}
                                </span>
                                <span style={{
                                    padding: '3px 10px', borderRadius: 20, fontSize: '0.7rem', fontWeight: 600,
                                    background: 'rgba(245, 158, 11, 0.15)', color: '#f59e0b',
                                }}>
                                    E = {bio.evenness.toFixed(3)}
                                </span> */}
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}
