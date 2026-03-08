import React, { Suspense } from 'react';
import { Canvas } from '@react-three/fiber';
import {
    OrbitControls,
    Stage,
    useGLTF,
    PerspectiveCamera,
    Environment,
    ContactShadows,
    Float,
    Grid
} from '@react-three/drei';
import { Download, Maximize2, RotateCcw, Box } from 'lucide-react';

interface ModelProps {
    url: string;
}

const Model = ({ url }: ModelProps) => {
    // For .vox files, we still want to use GLTFLoader since we exported them as GLB
    const { scene } = useGLTF(url, true); // true to use Draco if available, but here just ensures it loads
    return (
        <Float speed={2} rotationIntensity={0.5} floatIntensity={0.5}>
            <primitive object={scene} scale={1.5} position={[0, -0.5, 0]} />
        </Float>
    );
};

interface ViewerProps {
    assetUrl: string;
    voxelGridUrl?: string;
    radiographyUrl?: string;
    jobId: string;
    sourceImage?: string;
}

export const Viewer3D: React.FC<ViewerProps> = ({ assetUrl, voxelGridUrl, radiographyUrl, jobId, sourceImage }) => {
    const [viewMode, setViewMode] = React.useState<'3d' | 'radiography'>('3d');
    const fullUrl = `http://localhost:8000${assetUrl}`;
    const isGLB = assetUrl.toLowerCase().endsWith('.glb') || assetUrl.toLowerCase().endsWith('.vox');

    const handleDownload = () => {
        const extension = assetUrl.split('.').pop() || 'glb';
        const link = document.createElement('a');
        link.href = fullUrl;
        link.download = `3dgan_${jobId.slice(0, 8)}.${extension}`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    };

    const handleDownloadVoxels = () => {
        if (!voxelGridUrl) return;
        const fullVoxelUrl = `http://localhost:8000${voxelGridUrl}`;
        const link = document.createElement('a');
        link.href = fullVoxelUrl;
        link.download = `3dgan_${jobId.slice(0, 8)}_voxels.npy`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    };

    return (
        <div className="glass-container" style={{ minHeight: '600px', display: 'flex', flexDirection: 'column' }}>
            <div style={{ flex: 1, display: 'flex', position: 'relative' }}>
                {/* Source Input Representation */}
                {sourceImage && (
                    <div className="flex-col" style={{ width: '240px', borderRight: '1px solid var(--border-glass)', background: 'rgba(0,0,0,0.3)', padding: '1.5rem', display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                        <div className="flex-center gap-small" style={{ color: 'var(--text-dim)', fontSize: '0.65rem', fontWeight: 800, textTransform: 'uppercase', letterSpacing: '1px' }}>
                            <Box size={12} /> Source Input
                        </div>
                        <div style={{ flex: 1, borderRadius: '12px', overflow: 'hidden', border: '1px solid rgba(255,255,255,0.05)', position: 'relative' }}>
                            <img src={sourceImage} alt="Input" style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
                            <div style={{ position: 'absolute', inset: 0, background: 'linear-gradient(to bottom, transparent, rgba(0,0,0,0.5))' }}></div>
                        </div>
                        <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', lineHeight: 1.4 }}>
                            This image was used as the volumetric seed for the 3D generation.
                        </div>
                    </div>
                )}

                <div style={{ flex: 1, position: 'relative' }}>
                    {viewMode === '3d' ? (
                        isGLB ? (
                            <Canvas shadows gl={{ antialias: true, preserveDrawingBuffer: true }}>
                                <PerspectiveCamera makeDefault position={[3, 3, 5]} fov={45} />

                                <Suspense fallback={null}>
                                    <Stage intensity={0.5} environment="city" adjustCamera={false} shadows="contact">
                                        <Model url={fullUrl} />
                                    </Stage>

                                    <Environment preset="city" />
                                    <ContactShadows
                                        opacity={0.4}
                                        scale={10}
                                        blur={2.4}
                                        far={4.5}
                                        color="#000000"
                                    />

                                    <Grid
                                        infiniteGrid
                                        fadeDistance={20}
                                        fadeStrength={5}
                                        sectionSize={1}
                                        sectionThickness={1.5}
                                        sectionColor="rgba(99, 102, 241, 0.1)"
                                        cellColor="rgba(255, 255, 255, 0.02)"
                                    />
                                </Suspense>

                                <OrbitControls
                                    autoRotate
                                    autoRotateSpeed={2.5}  // Faster rotation for the "moving" effect
                                    enableDamping
                                    dampingFactor={0.05}
                                    minPolarAngle={Math.PI / 4}
                                    maxPolarAngle={Math.PI / 1.5}
                                />
                            </Canvas>
                        ) : (
                            <div className="flex-col flex-center" style={{ height: '100%', gap: '1.5rem', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center' }}>
                                <div className="glass-container" style={{ padding: '2rem', borderRadius: '50%', color: 'var(--accent-primary)', background: 'rgba(99, 102, 241, 0.05)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                                    <Box size={48} />
                                </div>
                                <div className="text-center">
                                    <h3 style={{ fontSize: '1.25rem', fontWeight: 800 }}>Preview Unsupported</h3>
                                    <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem', maxWidth: '300px', margin: '0.5rem auto' }}>
                                        QuickLook preview is only available for GLB/VOX. Please download the file to view in your 3D software.
                                    </p>
                                </div>
                            </div>
                        )
                    ) : (
                        <div className="flex-col flex-center" style={{ height: '100%', background: 'rgba(0,0,0,0.8)', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', padding: '2rem' }}>
                            <div style={{ position: 'relative', width: '80%', aspectRatio: '1/1', border: '2px dashed #a855f7', borderRadius: '12px', overflow: 'hidden', background: '#000' }}>
                                <img
                                    src={`http://localhost:8000${radiographyUrl}`}
                                    alt="Radiography"
                                    style={{ width: '100%', height: '100%', objectFit: 'contain', filter: 'brightness(1.5) contrast(1.2) invert(1)' }}
                                />
                                <div style={{ position: 'absolute', top: '10px', left: '10px', color: '#a855f7', fontSize: '0.7rem', fontWeight: 800 }}>X-RAY THICKNESS SCAN</div>
                            </div>
                            <p style={{ marginTop: '1rem', color: 'var(--text-secondary)', fontSize: '0.8rem', maxWidth: '400px', textAlign: 'center' }}>
                                Brighter areas indicate higher volumetric density and object thickness along the orthographic projection Z-axis.
                            </p>
                        </div>
                    )}

                    {/* Floating Controls Overlay */}
                    <div style={{ position: 'absolute', top: '1.5rem', right: '1.5rem', display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
                        <button className="flex-center" style={{ width: '40px', height: '40px', borderRadius: '12px', background: 'rgba(0,0,0,0.5)', backdropFilter: 'blur(100px)', border: '1px solid var(--border-glass)', color: 'white', cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                            <RotateCcw size={18} />
                        </button>
                        <button className="flex-center" style={{ width: '40px', height: '40px', borderRadius: '12px', background: 'rgba(0,0,0,0.5)', backdropFilter: 'blur(100px)', border: '1px solid var(--border-glass)', color: 'white', cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                            <Maximize2 size={18} />
                        </button>
                    </div>

                    <div style={{ position: 'absolute', bottom: '1.5rem', left: '1.5rem' }}>
                        <div className="flex-center gap-small" style={{ display: 'flex', gap: '8px' }}>
                            <div className="flex-center gap-small" style={{ background: 'rgba(0,0,0,0.5)', padding: '8px 16px', borderRadius: '10px', backdropFilter: 'blur(10px)', border: '1px solid var(--border-glass)', display: 'flex', alignItems: 'center', gap: '8px' }}>
                                <div style={{ width: '6px', height: '6px', borderRadius: '50%', background: viewMode === '3d' ? 'var(--accent-primary)' : '#a855f7' }}></div>
                                <span style={{ fontSize: '0.75rem', fontWeight: 700, letterSpacing: '0.5px' }}>{viewMode === '3d' ? '3D VIEWPORT' : 'THICKNESS RADIOGRAPHY'}</span>
                            </div>

                            {radiographyUrl && (
                                <button
                                    onClick={() => setViewMode(viewMode === '3d' ? 'radiography' : '3d')}
                                    className="glass-container flex-center"
                                    style={{ padding: '8px 16px', fontSize: '0.7rem', fontWeight: 800, background: 'rgba(168, 85, 247, 0.2)', border: '1px solid #a855f7', cursor: 'pointer', borderRadius: '10px' }}
                                >
                                    SWITCH TO {viewMode === '3d' ? 'ANALYSIS' : '3D VIEW'}
                                </button>
                            )}
                        </div>
                    </div>
                </div>
            </div>

            <div style={{ padding: '1.5rem', background: 'rgba(0,0,0,0.2)', backdropFilter: 'blur(10px)', borderTop: '1px solid var(--border-glass)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <div className="flex" style={{ display: 'flex', gap: '2rem' }}>
                    <div className="flex-col" style={{ display: 'flex', flexDirection: 'column' }}>
                        <span style={{ fontSize: '0.7rem', color: 'var(--text-dim)', textTransform: 'uppercase', letterSpacing: '1px' }}>Asset Format</span>
                        <span style={{ fontWeight: 800 }}>{(assetUrl.split('.').pop() || 'glb').toUpperCase()} Asset</span>
                    </div>
                    {voxelGridUrl && (
                        <div className="flex-col" style={{ display: 'flex', flexDirection: 'column' }}>
                            <span style={{ fontSize: '0.7rem', color: 'var(--text-dim)', textTransform: 'uppercase', letterSpacing: '1px' }}>Voxel Data</span>
                            <span style={{ fontWeight: 800 }}>RAW Grid (.npy)</span>
                        </div>
                    )}
                </div>

                <div className="flex gap-small" style={{ display: 'flex', gap: '0.75rem' }}>
                    {voxelGridUrl && (
                        <button onClick={handleDownloadVoxels} className="premium-button" style={{ background: 'rgba(255,255,255,0.05)', color: 'white' }}>
                            <Box size={18} /> Voxel Grid
                        </button>
                    )}
                    <button onClick={handleDownload} className="premium-button">
                        <Download size={18} /> Download 3D Asset
                    </button>
                </div>
            </div>
        </div>
    );
};
