import {
    BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
    AreaChart, Area, Cell, Legend,
} from 'recharts';

const CHART_COLORS = {
    green: '#10b981',
    teal: '#14b8a6',
    blue: '#3b82f6',
    amber: '#f59e0b',
    purple: '#a78bfa',
    rose: '#f43f5e',
};

const tooltipStyle = {
    backgroundColor: '#1a2332',
    border: '1px solid rgba(148, 163, 184, 0.2)',
    borderRadius: '10px',
    color: '#f1f5f9',
    fontSize: '0.8rem',
};

const tooltipLabelStyle = { color: '#f1f5f9' };
const tooltipItemStyle = { color: '#e2e8f0' };

/* ---- CSS Gauge Component ---- */
function DensityGauge({ density }) {
    const percentage = Math.round(density * 100);
    const rotation = -90 + (density * 180); // -90 to 90 degrees

    // Color based on density
    let color = '#10b981';
    if (density < 0.3) color = '#f59e0b';
    else if (density < 0.6) color = '#14b8a6';

    return (
        <div className="gauge-container">
            <div style={{ position: 'relative', width: 200, height: 110 }}>
                {/* Background arc */}
                <svg viewBox="0 0 200 110" style={{ width: 200, height: 110 }}>
                    <defs>
                        <linearGradient id="gaugeGrad" x1="0%" y1="0%" x2="100%" y2="0%">
                            <stop offset="0%" stopColor="#f59e0b" />
                            <stop offset="50%" stopColor="#14b8a6" />
                            <stop offset="100%" stopColor="#10b981" />
                        </linearGradient>
                    </defs>
                    {/* Background */}
                    <path
                        d="M 20 100 A 80 80 0 0 1 180 100"
                        fill="none"
                        stroke="#243042"
                        strokeWidth="14"
                        strokeLinecap="round"
                    />
                    {/* Filled arc */}
                    <path
                        d="M 20 100 A 80 80 0 0 1 180 100"
                        fill="none"
                        stroke="url(#gaugeGrad)"
                        strokeWidth="14"
                        strokeLinecap="round"
                        strokeDasharray={`${density * 251.2} 251.2`}
                        style={{ transition: 'stroke-dasharray 1.2s cubic-bezier(0.4, 0, 0.2, 1)' }}
                    />
                    {/* Needle */}
                    <line
                        x1="100" y1="100"
                        x2={100 + 60 * Math.cos((rotation * Math.PI) / 180)}
                        y2={100 + 60 * Math.sin((rotation * Math.PI) / 180)}
                        stroke={color}
                        strokeWidth="3"
                        strokeLinecap="round"
                        style={{ transition: 'all 1.2s cubic-bezier(0.4, 0, 0.2, 1)' }}
                    />
                    {/* Center dot */}
                    <circle cx="100" cy="100" r="5" fill={color} />
                </svg>
                <div style={{
                    position: 'absolute', bottom: 0, left: '50%', transform: 'translateX(-50%)',
                    fontSize: '1.6rem', fontWeight: 800, color,
                }}>
                    {percentage}%
                </div>
            </div>
            <div className="gauge-label">Tree Density Index</div>
        </div>
    );
}

export default function TreeAnalytics({ results }) {
    if (!results) return null;

    // Data for density breakdown
    const densityData = [
        { name: 'Density', value: results.density, fill: CHART_COLORS.green },
        { name: 'Remaining', value: 1 - results.density, fill: '#243042' },
    ];

    // Tree count breakdown
    const treeMetrics = [
        { name: 'Total Trees', value: results.tree_count, fill: CHART_COLORS.green },
        { name: 'Per Hectare', value: results.trees_per_hectare, fill: CHART_COLORS.teal },
        { name: 'Per km²', value: results.trees_per_sqkm, fill: CHART_COLORS.blue },
    ];

    // Confidence data
    const confidenceData = [
        { name: 'Average', value: (results.confidence_stats.average * 100).toFixed(1), fill: CHART_COLORS.amber },
        { name: 'Maximum', value: (results.confidence_stats.maximum * 100).toFixed(1), fill: CHART_COLORS.green },
        { name: 'Minimum', value: (results.confidence_stats.minimum * 100).toFixed(1), fill: CHART_COLORS.rose },
        { name: 'Std Dev', value: (results.confidence_stats.std_dev * 100).toFixed(1), fill: CHART_COLORS.purple },
    ];

    // Density distribution (simulated histogram from single prediction)
    const histData = [];
    for (let i = 0; i <= 10; i++) {
        const binCenter = i / 10;
        const dist = Math.exp(-Math.pow(binCenter - results.density, 2) / 0.05);
        histData.push({
            range: `${(i * 10)}%`,
            frequency: Math.round(dist * 100),
        });
    }

    return (
        <div className="dashboard-section animate-fade-in">
            <div className="section-header">
                <div className="section-icon green">🌲</div>
                <div>
                    <div className="section-title">Tree Count Analytics</div>
                    <div className="section-desc">Density estimation and tree count metrics</div>
                </div>
            </div>

            <div className="chart-grid">
                {/* Gauge Chart */}
                <div className="chart-card">
                    <div className="chart-card-title">
                        <span className="dot green"></span>Tree Density Gauge
                    </div>
                    <DensityGauge density={results.density} />
                </div>

                {/* Tree Metrics Bar Chart */}
                <div className="chart-card">
                    <div className="chart-card-title">
                        <span className="dot blue"></span>Tree Count Metrics
                    </div>
                    <ResponsiveContainer width="100%" height={250}>
                        <BarChart data={treeMetrics} margin={{ top: 5, right: 20, bottom: 5, left: 10 }}>
                            <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.1)" />
                            <XAxis
                                dataKey="name"
                                tick={{ fill: '#94a3b8', fontSize: 12 }}
                                axisLine={{ stroke: 'rgba(148,163,184,0.1)' }}
                            />
                            <YAxis
                                tick={{ fill: '#94a3b8', fontSize: 12 }}
                                axisLine={{ stroke: 'rgba(148,163,184,0.1)' }}
                            />
                            <Tooltip contentStyle={tooltipStyle} labelStyle={tooltipLabelStyle} itemStyle={tooltipItemStyle} />
                            <Bar dataKey="value" radius={[6, 6, 0, 0]} barSize={50}>
                                {treeMetrics.map((entry, i) => (
                                    <Cell key={i} fill={entry.fill} />
                                ))}
                            </Bar>
                        </BarChart>
                    </ResponsiveContainer>
                </div>

                {/* Confidence Stats */}
                <div className="chart-card">
                    <div className="chart-card-title">
                        <span className="dot amber"></span>Confidence Statistics
                    </div>
                    <ResponsiveContainer width="100%" height={250}>
                        <BarChart data={confidenceData} layout="vertical" margin={{ top: 5, right: 20, bottom: 5, left: 60 }}>
                            <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.1)" />
                            <XAxis
                                type="number"
                                domain={[0, 100]}
                                tick={{ fill: '#94a3b8', fontSize: 12 }}
                                axisLine={{ stroke: 'rgba(148,163,184,0.1)' }}
                                unit="%"
                            />
                            <YAxis
                                type="category"
                                dataKey="name"
                                tick={{ fill: '#94a3b8', fontSize: 12 }}
                                axisLine={{ stroke: 'rgba(148,163,184,0.1)' }}
                            />
                            <Tooltip contentStyle={tooltipStyle} labelStyle={tooltipLabelStyle} itemStyle={tooltipItemStyle} formatter={(v) => `${v}%`} />
                            <Bar dataKey="value" radius={[0, 6, 6, 0]} barSize={24}>
                                {confidenceData.map((entry, i) => (
                                    <Cell key={i} fill={entry.fill} />
                                ))}
                            </Bar>
                        </BarChart>
                    </ResponsiveContainer>
                </div>

                {/* Density Distribution Histogram */}
                <div className="chart-card">
                    <div className="chart-card-title">
                        <span className="dot green"></span>Density Distribution
                    </div>
                    <ResponsiveContainer width="100%" height={250}>
                        <AreaChart data={histData} margin={{ top: 5, right: 20, bottom: 5, left: 10 }}>
                            <defs>
                                <linearGradient id="densityGrad" x1="0" y1="0" x2="0" y2="1">
                                    <stop offset="0%" stopColor="#10b981" stopOpacity={0.4} />
                                    <stop offset="100%" stopColor="#10b981" stopOpacity={0.05} />
                                </linearGradient>
                            </defs>
                            <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.1)" />
                            <XAxis
                                dataKey="range"
                                tick={{ fill: '#94a3b8', fontSize: 11 }}
                                axisLine={{ stroke: 'rgba(148,163,184,0.1)' }}
                            />
                            <YAxis
                                tick={{ fill: '#94a3b8', fontSize: 12 }}
                                axisLine={{ stroke: 'rgba(148,163,184,0.1)' }}
                            />
                            <Tooltip contentStyle={tooltipStyle} labelStyle={tooltipLabelStyle} itemStyle={tooltipItemStyle} />
                            <Area
                                type="monotone"
                                dataKey="frequency"
                                stroke="#10b981"
                                fill="url(#densityGrad)"

                                strokeWidth={2}
                            />
                        </AreaChart>
                    </ResponsiveContainer>
                </div>
            </div>
        </div>
    );
}
