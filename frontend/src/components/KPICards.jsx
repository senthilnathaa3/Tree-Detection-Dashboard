import { useEffect, useState, useRef } from 'react';

/* Animated counter that counts up from 0 to target value */
function AnimatedNumber({ value, decimals = 0, duration = 1200, prefix = '', suffix = '' }) {
    const [display, setDisplay] = useState(0);
    const frameRef = useRef();
    const startRef = useRef();

    useEffect(() => {
        if (value === null || value === undefined) return;
        const target = parseFloat(value);
        if (isNaN(target)) return;

        const startVal = 0;
        startRef.current = performance.now();

        const animate = (timestamp) => {
            const elapsed = timestamp - startRef.current;
            const progress = Math.min(elapsed / duration, 1);
            // Ease out cubic
            const eased = 1 - Math.pow(1 - progress, 3);
            const current = startVal + (target - startVal) * eased;
            setDisplay(current);

            if (progress < 1) {
                frameRef.current = requestAnimationFrame(animate);
            }
        };

        frameRef.current = requestAnimationFrame(animate);
        return () => cancelAnimationFrame(frameRef.current);
    }, [value, duration]);

    const formatted = typeof value === 'number'
        ? display.toFixed(decimals)
        : value;

    return (
        <span>
            {prefix}{typeof value === 'number' ? Number(display.toFixed(decimals)).toLocaleString() : formatted}{suffix}
        </span>
    );
}

export default function KPICards({ results }) {
    if (!results) return null;

    const cards = [
        {
            icon: '🌲',
            label: 'Estimated Trees',
            value: results.tree_count,
            decimals: 0,
            color: 'green',
            subtext: `In ${results.patch_area_hectares} ha patch`,
        },
        {
            icon: '📊',
            label: 'Trees per Hectare',
            value: results.trees_per_hectare,
            decimals: 1,
            color: 'teal',
            subtext: 'Density metric',
        },
        {
            icon: '🗺️',
            label: 'Trees per km²',
            value: results.trees_per_sqkm,
            decimals: 0,
            color: 'blue',
            subtext: 'Regional density',
        },
        {
            icon: '🌿',
            label: 'Dominant Species',
            value: results.dominant_species,
            color: 'green',
            subtext: `${results.total_species_detected} species detected`,
            isText: true,
        },
        // {
        //     icon: '🎯',
        //     label: 'Avg Confidence',
        //     value: results.confidence_stats?.average * 100,
        //     decimals: 1,
        //     suffix: '%',
        //     color: 'amber',
        //     subtext: `Min: ${(results.confidence_stats?.minimum * 100).toFixed(1)}%`,
        // },
        {
            icon: '🧬',
            label: 'Biodiversity Score',
            value: results.biodiversity_metrics?.biodiversity_score,
            decimals: 1,
            suffix: '%',
            color: 'purple',
            subtext: `Shannon: ${results.biodiversity_metrics?.shannon_index}`,
        },
    ];

    return (
        <div className="kpi-row">
            {cards.map((card, i) => (
                <div
                    key={card.label}
                    className={`kpi-card ${card.color} animate-fade-in delay-${i + 1}`}
                >
                    <div className="kpi-icon">{card.icon}</div>
                    <div className="kpi-label">{card.label}</div>
                    <div className={`kpi-value ${card.color}`}>
                        {card.isText ? (
                            card.value
                        ) : (
                            <AnimatedNumber
                                value={card.value}
                                decimals={card.decimals || 0}
                                prefix={card.prefix || ''}
                                suffix={card.suffix || ''}
                            />
                        )}
                    </div>
                    <div className="kpi-subtext">{card.subtext}</div>
                </div>
            ))}
        </div>
    );
}
