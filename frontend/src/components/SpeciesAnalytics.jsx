import {
    PieChart, Pie, Cell, ResponsiveContainer, Tooltip,
    BarChart, Bar, XAxis, YAxis, CartesianGrid,
    RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar,
    Legend,
} from 'recharts';

const SPECIES_COLORS = [
    '#10b981', '#14b8a6', '#0ea5e9', '#3b82f6', '#6366f1',
    '#8b5cf6', '#a78bfa', '#d946ef', '#ec4899', '#f43f5e',
    '#f59e0b', '#eab308', '#84cc16', '#22c55e', '#06b6d4',
    '#0891b2', '#7c3aed', '#c026d3', '#e11d48', '#ea580c',
];

const tooltipStyle = {
    backgroundColor: '#1a2332',
    border: '1px solid rgba(148, 163, 184, 0.2)',
    borderRadius: '10px',
    color: '#f1f5f9',
    fontSize: '0.8rem',
};

const tooltipLabelStyle = { color: '#f1f5f9' };
const tooltipItemStyle = { color: '#e2e8f0' };

/* Custom pie label */
function renderCustomLabel({ cx, cy, midAngle, innerRadius, outerRadius, percent, name }) {
    if (percent < 0.05) return null;
    const RADIAN = Math.PI / 180;
    const radius = innerRadius + (outerRadius - innerRadius) * 1.4;
    const x = cx + radius * Math.cos(-midAngle * RADIAN);
    const y = cy + radius * Math.sin(-midAngle * RADIAN);

    return (
        <text
            x={x} y={y}
            fill="#94a3b8"
            textAnchor={x > cx ? 'start' : 'end'}
            dominantBaseline="central"
            fontSize={11}
        >
            {name} ({(percent * 100).toFixed(0)}%)
        </text>
    );
}

export default function SpeciesAnalytics({ results }) {
    if (!results) return null;

    const { species_distribution } = results;

    // Pie chart data (top species)
    const pieData = species_distribution
        .filter(s => s.probability > 0.02)
        .map(s => ({
            name: s.species,
            value: Math.round(s.probability * 1000) / 10,
        }));

    // Bar chart data (top 10)
    const top10 = species_distribution.slice(0, 10).map(s => ({
        species: s.species,
        probability: Math.round(s.probability * 100),
        detected: s.detected,
    }));

    // Horizontal ranking
    const rankingData = [...species_distribution]
        .sort((a, b) => b.probability - a.probability)
        .map((s, i) => ({
            rank: i + 1,
            species: s.species,
            probability: Math.round(s.probability * 100),
        }));

    // Radar chart data (top 8 for readability)
    const radarData = species_distribution.slice(0, 8).map(s => ({
        species: s.species,
        probability: Math.round(s.probability * 100),
        fullMark: 100,
    }));

    // Confidence histogram bins
    const confBins = [
        { range: '0-20%', count: 0 },
        { range: '20-40%', count: 0 },
        { range: '40-60%', count: 0 },
        { range: '60-80%', count: 0 },
        { range: '80-100%', count: 0 },
    ];
    species_distribution.forEach(s => {
        const p = s.probability * 100;
        if (p < 20) confBins[0].count++;
        else if (p < 40) confBins[1].count++;
        else if (p < 60) confBins[2].count++;
        else if (p < 80) confBins[3].count++;
        else confBins[4].count++;
    });

    const detectedCount = species_distribution.filter(s => s.detected).length;

    return (
        <div className="dashboard-section animate-fade-in">
            <div className="section-header">
                <div className="section-icon teal">🌿</div>
                <div>
                    <div className="section-title">Species Analytics</div>
                    <div className="section-desc">
                        {detectedCount} of {species_distribution.length} species detected •
                        Dominant: <strong style={{ color: '#10b981' }}>{results.dominant_species}</strong>
                    </div>
                </div>
            </div>

            <div className="chart-grid">
                {/* Pie Chart */}
                <div className="chart-card">
                    <div className="chart-card-title">
                        <span className="dot green"></span>Species Distribution
                    </div>
                    <ResponsiveContainer width="100%" height={320}>
                        <PieChart>
                            <Pie
                                data={pieData}
                                cx="50%"
                                cy="50%"
                                innerRadius={60}
                                outerRadius={100}
                                paddingAngle={2}
                                dataKey="value"
                                label={renderCustomLabel}
                                animationBegin={200}
                                animationDuration={1200}
                            >
                                {pieData.map((_, i) => (
                                    <Cell key={i} fill={SPECIES_COLORS[i % SPECIES_COLORS.length]} />
                                ))}
                            </Pie>
                            <Tooltip contentStyle={tooltipStyle} labelStyle={tooltipLabelStyle} itemStyle={tooltipItemStyle} formatter={(v) => `${v}%`} />
                        </PieChart>
                    </ResponsiveContainer>
                </div>

                {/* Top 10 Bar Chart */}
                <div className="chart-card">
                    <div className="chart-card-title">
                        <span className="dot blue"></span>Top 10 Species Probabilities
                    </div>
                    <ResponsiveContainer width="100%" height={320}>
                        <BarChart data={top10} layout="horizontal" margin={{ top: 5, right: 20, bottom: 5, left: 0 }}>
                            <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.1)" />
                            <XAxis
                                type="category"
                                dataKey="species"

                                domain={[0, 100]}
                                tick={{ fill: '#94a3b8', fontSize: 12 }}
                                axisLine={{ stroke: 'rgba(148,163,184,0.1)' }}
                                unit="%"
                            />
                            <YAxis
                                type="number"
                                tick={{ fill: '#94a3b8', fontSize: 12 }}
                                axisLine={{ stroke: 'rgba(148,163,184,0.1)' }}
                                width={75}
                            />
                            <Tooltip contentStyle={tooltipStyle} labelStyle={tooltipLabelStyle} itemStyle={tooltipItemStyle} formatter={(v) => `${v}%`} />
                            <Bar dataKey="probability" radius={[0, 6, 6, 0]} barSize={18}>
                                {top10.map((entry, i) => (
                                    <Cell key={i} fill={entry.detected ? SPECIES_COLORS[i] : 'rgba(148,163,184,0.3)'} />
                                ))}
                            </Bar>
                        </BarChart>
                    </ResponsiveContainer>
                </div>

                {/* Radar Chart */}
                <div className="chart-card">
                    <div className="chart-card-title">
                        <span className="dot purple"></span>Species Dominance Profile
                    </div>
                    <ResponsiveContainer width="100%" height={320}>
                        <RadarChart data={radarData}>
                            <PolarGrid stroke="rgba(148,163,184,0.15)" />
                            <PolarAngleAxis
                                dataKey="species"
                                tick={{ fill: '#94a3b8', fontSize: 11 }}
                            />
                            <PolarRadiusAxis
                                angle={90}
                                domain={[0, 100]}
                                tick={{ fill: '#64748b', fontSize: 10 }}
                                axisLine={false}
                            />
                            <Radar
                                dataKey="probability"
                                stroke="#10b981"
                                fill="#10b981"
                                fillOpacity={0.2}
                                strokeWidth={2}
                                animationDuration={1200}
                            />
                            <Tooltip contentStyle={tooltipStyle} labelStyle={tooltipLabelStyle} itemStyle={tooltipItemStyle} formatter={(v) => `${v}%`} />
                        </RadarChart>
                    </ResponsiveContainer>
                </div>

                {/* Confidence Histogram */}
                <div className="chart-card">
                    <div className="chart-card-title">
                        <span className="dot amber"></span>Confidence Distribution
                    </div>
                    <ResponsiveContainer width="100%" height={320}>
                        <BarChart data={confBins} margin={{ top: 5, right: 20, bottom: 5, left: 10 }}>
                            <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.1)" />
                            <XAxis
                                dataKey="range"
                                tick={{ fill: '#94a3b8', fontSize: 12 }}
                                axisLine={{ stroke: 'rgba(148,163,184,0.1)' }}
                            />
                            <YAxis
                                tick={{ fill: '#94a3b8', fontSize: 12 }}
                                axisLine={{ stroke: 'rgba(148,163,184,0.1)' }}
                                label={{ value: 'Species Count', angle: -90, position: 'insideLeft', fill: '#64748b', fontSize: 11 }}
                            />
                            <Tooltip contentStyle={tooltipStyle} labelStyle={tooltipLabelStyle} itemStyle={tooltipItemStyle} />
                            <Bar dataKey="count" radius={[6, 6, 0, 0]} barSize={40}>
                                {confBins.map((_, i) => (
                                    <Cell key={i} fill={SPECIES_COLORS[i * 3]} />
                                ))}
                            </Bar>
                        </BarChart>
                    </ResponsiveContainer>
                </div>
            </div>
        </div>
    );
}
