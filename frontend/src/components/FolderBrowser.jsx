import { useState, useEffect, useCallback } from 'react';
import { browseDirectory } from '../services/api';

export default function FolderBrowser({ isOpen, onClose, onSelect }) {
    const [currentPath, setCurrentPath] = useState('');
    const [parentPath, setParentPath] = useState('');
    const [entries, setEntries] = useState([]);
    const [isDataset, setIsDataset] = useState(false);
    const [hasS1, setHasS1] = useState(false);
    const [hasS2, setHasS2] = useState(false);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    const browse = useCallback(async (path) => {
        setLoading(true);
        setError(null);
        try {
            const data = await browseDirectory(path);
            setCurrentPath(data.current_path);
            setParentPath(data.parent_path);
            setEntries(data.entries || []);
            setIsDataset(data.is_dataset);
            setHasS1(data.has_s1);
            setHasS2(data.has_s2);
        } catch (err) {
            setError(err.message);
        } finally {
            setLoading(false);
        }
    }, []);

    // Load home directory when opened
    useEffect(() => {
        if (isOpen) {
            browse('~');
        }
    }, [isOpen, browse]);

    const handleSelect = useCallback(() => {
        onSelect(currentPath);
        onClose();
    }, [currentPath, onSelect, onClose]);

    // Close on escape key
    useEffect(() => {
        const handleKey = (e) => {
            if (e.key === 'Escape' && isOpen) onClose();
        };
        window.addEventListener('keydown', handleKey);
        return () => window.removeEventListener('keydown', handleKey);
    }, [isOpen, onClose]);

    if (!isOpen) return null;

    return (
        <div className="folder-browser-overlay" onClick={onClose}>
            <div className="folder-browser-modal" onClick={(e) => e.stopPropagation()}>
                {/* Header */}
                <div className="folder-browser-header">
                    <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                        <span style={{ fontSize: '1.3rem' }}>📂</span>
                        <div>
                            <div className="folder-browser-title">Select Dataset Folder</div>
                            <div className="folder-browser-subtitle">
                                Choose a directory with <code>s1/</code> and <code>s2/</code> subdirectories
                            </div>
                        </div>
                    </div>
                    <button className="folder-browser-close" onClick={onClose}>✕</button>
                </div>

                {/* Breadcrumb / Current Path */}
                <div className="folder-browser-path">
                    <button
                        className="folder-browser-up-btn"
                        onClick={() => browse(parentPath)}
                        disabled={currentPath === parentPath || loading}
                        title="Go to parent directory"
                    >
                        ⬆️
                    </button>
                    <div className="folder-browser-path-text">{currentPath}</div>
                </div>

                {/* Dataset Indicator */}
                {(hasS1 || hasS2) && (
                    <div className={`folder-browser-dataset-indicator ${isDataset ? 'valid' : 'partial'}`}>
                        <span style={{ fontSize: '1rem' }}>{isDataset ? '✅' : '⚠️'}</span>
                        <div>
                            <div style={{ fontWeight: 600, fontSize: '0.82rem' }}>
                                {isDataset ? 'Valid dataset directory' : 'Partial dataset directory'}
                            </div>
                            <div style={{ fontSize: '0.72rem', opacity: 0.8 }}>
                                {hasS1 ? '✓ s1/' : '✗ s1/'} &nbsp;
                                {hasS2 ? '✓ s2/' : '✗ s2/'}
                            </div>
                        </div>
                    </div>
                )}

                {/* Error */}
                {error && (
                    <div style={{
                        padding: '8px 14px',
                        margin: '0 16px 8px',
                        fontSize: '0.78rem',
                        color: '#f43f5e',
                        background: 'rgba(244,63,94,0.1)',
                        borderRadius: 8,
                    }}>
                        {error}
                    </div>
                )}

                {/* Directory List */}
                <div className="folder-browser-list">
                    {loading ? (
                        <div className="folder-browser-loading">
                            <div className="spinner" style={{ width: 24, height: 24, borderWidth: 2 }}></div>
                            <span>Loading...</span>
                        </div>
                    ) : entries.length === 0 ? (
                        <div className="folder-browser-empty">
                            No subdirectories found
                        </div>
                    ) : (
                        entries.map((entry) => (
                            <button
                                key={entry.path}
                                className={`folder-browser-entry ${entry.name === 's1' || entry.name === 's2' ? 'highlight' : ''}`}
                                onClick={() => browse(entry.path)}
                                title={entry.path}
                            >
                                <span className="folder-browser-entry-icon">
                                    {entry.name === 's1' || entry.name === 's2' ? '🛰️' : '📁'}
                                </span>
                                <div className="folder-browser-entry-info">
                                    <div className="folder-browser-entry-name">{entry.name}</div>
                                    {entry.tif_count > 0 && (
                                        <div className="folder-browser-entry-count">
                                            {entry.tif_count} .tif file{entry.tif_count !== 1 ? 's' : ''}
                                        </div>
                                    )}
                                </div>
                                <span className="folder-browser-entry-arrow">›</span>
                            </button>
                        ))
                    )}
                </div>

                {/* Footer with Select button */}
                <div className="folder-browser-footer">
                    <button className="btn btn-secondary" onClick={onClose}>
                        Cancel
                    </button>
                    <button
                        className="btn btn-primary"
                        onClick={handleSelect}
                        disabled={!isDataset}
                    >
                        {isDataset ? '✓ Select This Folder' : 'Select (requires s1/ & s2/)'}
                    </button>
                </div>
            </div>
        </div>
    );
}
