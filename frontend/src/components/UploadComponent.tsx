import React, { useCallback, useState } from 'react';
import { Upload, FileImage, ShieldCheck, Box } from 'lucide-react';

interface Props {
    onUpload: (file: File, format: string, skinRemovalLayers?: number) => void;
    disabled?: boolean;
}

export const UploadComponent: React.FC<Props> = ({ onUpload, disabled }) => {
    const [isDragging, setIsDragging] = useState(false);
    const [selectedFormat, setSelectedFormat] = useState('glb');
    const [skinRemovalLayers, setSkinRemovalLayers] = useState(0);

    const handleDrag = useCallback((e: React.DragEvent) => {
        e.preventDefault();
        e.stopPropagation();
        if (e.type === 'dragenter' || e.type === 'dragover') {
            setIsDragging(true);
        } else if (e.type === 'dragleave') {
            setIsDragging(false);
        }
    }, []);

    const handleDrop = useCallback((e: React.DragEvent) => {
        e.preventDefault();
        e.stopPropagation();
        setIsDragging(false);

        if (disabled) return;

        const file = e.dataTransfer.files?.[0];
        if (file && file.type.startsWith('image/')) {
            onUpload(file, selectedFormat, skinRemovalLayers);
        }
    }, [onUpload, disabled, selectedFormat, skinRemovalLayers]);

    const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0];
        if (file && !disabled) {
            onUpload(file, selectedFormat, skinRemovalLayers);
        }
    };

    const formats = [
        { id: 'glb', label: 'GLB', desc: 'Standard 3D (Web/AR)' },
        { id: 'obj', label: 'OBJ', desc: 'Wavefront Mesh' },
        { id: 'stl', label: 'STL', desc: '3D Printing' },
        { id: 'vox', label: 'VOX', desc: 'Voxelized Mesh' }
    ];

    return (
        <div className="flex-col flex-center w-full" style={{ maxWidth: '640px', display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
            <div
                className={`glass-container ${isDragging ? 'glow-hover' : ''}`}
                style={{
                    padding: '3rem',
                    borderStyle: isDragging ? 'solid' : 'dashed',
                    borderColor: isDragging ? 'var(--accent-primary)' : 'var(--border-glass)',
                    background: isDragging ? 'rgba(99, 102, 241, 0.08)' : 'var(--bg-card)',
                    transition: 'all 0.3s ease',
                    position: 'relative',
                    opacity: disabled ? 0.6 : 1,
                    pointerEvents: disabled ? 'none' : 'auto',
                    width: '100%'
                }}
                onDragEnter={handleDrag}
                onDragLeave={handleDrag}
                onDragOver={handleDrag}
                onDrop={handleDrop}
            >
                {/* Decorative Scanners for Dragging State */}
                {isDragging && (
                    <div style={{ position: 'absolute', inset: 0, pointerEvents: 'none', borderRadius: '28px', overflow: 'hidden' }}>
                        <div style={{ position: 'absolute', top: 0, left: 0, right: 0, height: '2px', background: 'var(--accent-primary)', boxShadow: '0 0 15px var(--accent-primary)', animation: 'scan 2s linear infinite' }}></div>
                    </div>
                )}

                <input
                    type="file"
                    id="file-upload"
                    accept="image/*"
                    onChange={handleChange}
                    disabled={disabled}
                    style={{ display: 'none' }}
                />

                <label
                    htmlFor="file-upload"
                    className="flex-col flex-center cursor-pointer"
                    style={{ gap: '1.5rem', display: 'flex', flexDirection: 'column', alignItems: 'center' }}
                >
                    <div className="flex-center" style={{
                        width: '80px',
                        height: '80px',
                        borderRadius: '24px',
                        background: 'rgba(255,255,255,0.03)',
                        color: isDragging ? 'var(--accent-primary)' : 'var(--text-secondary)',
                        transition: 'all 0.3s ease',
                        transform: isDragging ? 'scale(1.1) rotate(5deg)' : 'none',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center'
                    }}>
                        <Upload size={32} />
                    </div>

                    <div className="flex-col flex-center text-center" style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
                        <h3 style={{ fontSize: '1.25rem', fontWeight: 800, marginBottom: '0.5rem' }}>
                            {isDragging ? 'Drop Image Here' : 'Upload Source Image'}
                        </h3>
                        <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem', maxWidth: '280px' }}>
                            Drag and drop your high-resolution PNG or JPG to begin 3D reconstruction.
                        </p>
                    </div>
                </label>
            </div>

            {/* Format Selector */}
            <div style={{ marginTop: '2rem', width: '100%' }}>
                <p style={{ fontSize: '0.75rem', fontWeight: 800, color: 'var(--text-dim)', textTransform: 'uppercase', letterSpacing: '1.5px', marginBottom: '1rem', textAlign: 'center' }}>Select Output Architecture</p>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '1rem' }}>
                    {formats.map((f) => (
                        <button
                            key={f.id}
                            onClick={() => setSelectedFormat(f.id)}
                            disabled={disabled}
                            className={`glass-container ${selectedFormat === f.id ? 'glow-hover' : ''}`}
                            style={{
                                padding: '1rem',
                                textAlign: 'left',
                                border: selectedFormat === f.id ? '1px solid var(--accent-primary)' : '1px solid var(--border-glass)',
                                background: selectedFormat === f.id ? 'rgba(99, 102, 241, 0.05)' : 'rgba(0,0,0,0.2)',
                                cursor: 'pointer',
                                transition: 'all 0.2s ease',
                                borderRadius: '16px'
                            }}
                        >
                            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '4px' }}>
                                <Box size={14} style={{ color: selectedFormat === f.id ? 'var(--accent-primary)' : 'var(--text-dim)' }} />
                                <span style={{ fontWeight: 800, fontSize: '0.9rem', color: selectedFormat === f.id ? 'white' : 'var(--text-secondary)' }}>{f.label}</span>
                            </div>
                            <p style={{ fontSize: '0.7rem', color: 'var(--text-dim)' }}>{f.desc}</p>
                        </button>
                    ))}
                </div>
            </div>

            {/* Voxel Analysis Advanced Controls */}
            {selectedFormat === 'vox' && (
                <div className="glass-container mt-medium w-full" style={{ marginTop: '1.5rem', padding: '1.5rem', border: '1px solid rgba(168, 85, 247, 0.2)', background: 'rgba(168, 85, 247, 0.02)' }}>
                    <div className="flex items-center gap-2 mb-4">
                        <ShieldCheck size={16} className="text-accent-purple" color="#a855f7" />
                        <span style={{ fontSize: '0.8rem', fontWeight: 800, color: 'white', textTransform: 'uppercase', letterSpacing: '1px' }}>Voxel Analysis (Advanced)</span>
                    </div>

                    <div className="flex flex-col gap-4">
                        <div className="flex justify-between items-center">
                            <label style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>Skin Removal Layers</label>
                            <span style={{ fontSize: '0.9rem', fontWeight: 800, color: 'var(--accent-primary)' }}>{skinRemovalLayers}</span>
                        </div>
                        <input
                            type="range"
                            min="0"
                            max="5"
                            step="1"
                            value={skinRemovalLayers}
                            onChange={(e) => setSkinRemovalLayers(parseInt(e.target.value))}
                            style={{ width: '100%', accentColor: 'var(--accent-primary)', cursor: 'pointer' }}
                        />
                        <p style={{ fontSize: '0.7rem', color: 'var(--text-dim)', fontStyle: 'italic' }}>
                            Removes outer voxel shells to reveal internal thickness and density (Based on IIT Bombay research paper).
                        </p>
                    </div>
                </div>
            )}

            <div className="flex-center gap-medium" style={{ marginTop: '2rem', width: '100%', padding: '12px', background: 'rgba(0,0,0,0.2)', borderRadius: '12px', border: '1px solid rgba(255,255,255,0.02)', display: 'flex', justifyContent: 'center', gap: '20px' }}>
                <div className="flex-center gap-small" style={{ color: 'var(--text-dim)', fontSize: '0.75rem', display: 'flex', gap: '8px', alignItems: 'center' }}>
                    <ShieldCheck size={14} /> <span>Secure Upload</span>
                </div>
                <div className="flex-center gap-small" style={{ color: 'var(--text-dim)', fontSize: '0.75rem', display: 'flex', gap: '8px', alignItems: 'center' }}>
                    <FileImage size={14} /> <span>Max 10MB</span>
                </div>
            </div>
        </div>
    );
};
