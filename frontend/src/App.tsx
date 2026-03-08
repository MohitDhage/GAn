import { useState, Suspense } from 'react';
import { UploadComponent } from './components/UploadComponent';
import { ProgressPanel } from './components/ProgressPanel';
import { Viewer3D } from './components/Viewer3D';
import { apiClient } from './api/client';
import type { JobDetailResponse, JobStatusResponse } from './api/types';
import { Sparkles, Ruler, Zap, Box, History } from 'lucide-react';
import { useMeasurement, usePhysicsOverlay } from './hooks/future-features';

function App() {
  const [currentJob, setCurrentJob] = useState<JobDetailResponse | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [sourceImage, setSourceImage] = useState<string | null>(null);

  // Initialize future-proofing hooks
  const { startMeasurement } = useMeasurement();
  const { togglePhysics } = usePhysicsOverlay();

  const handleUpload = async (file: File, format: string, skinRemovalLayers: number = 0) => {
    try {
      setIsSubmitting(true);
      const imageUrl = URL.createObjectURL(file);
      setSourceImage(imageUrl);

      // 1. Submit job
      const initialResponse = await apiClient.submitGenerationJob(file, format, skinRemovalLayers);

      // 2. Initial state in UI
      const initialJob: JobDetailResponse = {
        job_id: initialResponse.job_id,
        status: initialResponse.status,
        progress: 0,
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString(),
      };
      setCurrentJob(initialJob);
      setIsSubmitting(false);

      // 3. Start polling
      const finalJob = await apiClient.pollJobUntilComplete(
        initialResponse.job_id,
        (statusUpdate: JobStatusResponse) => {
          setCurrentJob(prev => prev ? ({
            ...prev,
            status: statusUpdate.status,
            progress: statusUpdate.progress,
            updated_at: new Date().toISOString()
          }) : null);
        }
      );

      // 4. Update with final details (asset URL, etc)
      setCurrentJob(finalJob);

    } catch (error) {
      console.error('Generation failed:', error);
      setIsSubmitting(false);
      setCurrentJob(prev => prev ? ({ ...prev, status: 'FAILED', error_message: 'Network or system error' }) : null);
    }
  };

  const handleReset = () => {
    setCurrentJob(null);
    setIsSubmitting(false);
    setSourceImage(null);
  };

  return (
    <div className="p-section">
      {/* Decorative Blur Backgrounds */}
      <div style={{ position: 'fixed', top: '10%', right: '5%', width: '400px', height: '400px', background: 'rgba(99, 102, 241, 0.08)', filter: 'blur(100px)', borderRadius: '50%', zIndex: -1 }}></div>
      <div style={{ position: 'fixed', bottom: '10%', left: '5%', width: '300px', height: '300px', background: 'rgba(168, 85, 247, 0.08)', filter: 'blur(100px)', borderRadius: '50%', zIndex: -1 }}></div>

      <header className="max-w-container flex-col flex-center gap-medium" style={{ marginBottom: '5rem' }}>
        <div className="flex-center gap-small" style={{ background: 'rgba(255,255,255,0.03)', padding: '8px 16px', borderRadius: '40px', border: '1px solid rgba(255,255,255,0.05)' }}>
          <div style={{ width: '8px', height: '8px', borderRadius: '50%', background: '#10b981', boxShadow: '0 0 10px #10b981' }}></div>
          <span style={{ fontSize: '0.75rem', fontWeight: 800, color: 'rgba(255,255,255,0.6)', letterSpacing: '1px', textTransform: 'uppercase' }}>System Online : v1.0.4</span>
        </div>

        <div className="flex-col flex-center">
          <h1 className="title-gradient" style={{ fontSize: 'clamp(2.5rem, 8vw, 4.5rem)', textAlign: 'center', lineHeight: 1.1 }}>
            Visonary<br />3D Intelligence
          </h1>
          <p className="subtext" style={{ marginTop: '1.5rem', textAlign: 'center', maxWidth: '600px' }}>
            Unleash high-fidelity 3D assets from static 2D images.
            Bridging the gap between pixels and polygons.
          </p>
        </div>

        <div className="flex-center gap-medium" style={{ marginTop: '1rem' }}>
          <div className="flex-center gap-small" style={{ color: 'var(--text-dim)', fontSize: '0.85rem' }}>
            <Box size={16} /> <span>TripoSR Core</span>
          </div>
          <div className="flex-center gap-small" style={{ color: 'var(--text-dim)', fontSize: '0.85rem' }}>
            <Zap size={16} /> <span>RTX Optimized</span>
          </div>
        </div>
      </header>

      <main className="max-w-container">
        {!currentJob ? (
          <div className="flex-col flex-center">
            <UploadComponent onUpload={handleUpload} disabled={isSubmitting} />
            <div style={{ marginTop: '4rem', display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '2rem', width: '100%' }}>
              {[
                { icon: <Sparkles />, title: "Fully Textured", desc: "Realistic materials baked in." },
                { icon: <History />, title: "Real-time Poll", desc: "Instant progress monitoring." },
                { icon: <Box />, title: "GLB Export", desc: "Ready for Unity or Blender." }
              ].map((feature, i) => (
                <div key={i} className="glass-container" style={{ padding: '1.5rem', background: 'transparent' }}>
                  <div style={{ color: 'var(--accent-primary)', marginBottom: '1rem' }}>{feature.icon}</div>
                  <h3 style={{ fontSize: '1rem', fontWeight: 700, marginBottom: '0.5rem' }}>{feature.title}</h3>
                  <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>{feature.desc}</p>
                </div>
              ))}
            </div>
          </div>
        ) : (
          <div className="flex-col gap-medium">
            <div style={{ maxWidth: '700px', margin: '0 auto', width: '100%' }}>
              <ProgressPanel job={currentJob} onReset={handleReset} />
            </div>

            {currentJob.status === 'COMPLETED' && currentJob.asset_url && (
              <div className="flex-col gap-medium">
                <Suspense fallback={
                  <div className="glass-container flex-center" style={{ height: '500px', animation: 'pulse-glow 2s infinite' }}>
                    <div className="flex-col flex-center gap-small">
                      <div className="flex-center" style={{ border: '3px solid var(--accent-primary)', borderTopColor: 'transparent', width: '40px', height: '40px', borderRadius: '50%', animation: 'scan 1.5s linear infinite' }}></div>
                      <span style={{ fontSize: '0.9rem', color: 'var(--text-secondary)' }}>Initializing 3D Engine...</span>
                    </div>
                  </div>
                }>
                  <Viewer3D
                    assetUrl={currentJob.asset_url}
                    voxelGridUrl={currentJob.voxel_grid_url}
                    radiographyUrl={currentJob.radiography_url}
                    jobId={currentJob.job_id}
                    sourceImage={sourceImage || undefined}
                  />
                </Suspense>

                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.5rem' }}>
                  <button
                    onClick={startMeasurement}
                    className="glass-container glow-hover"
                    style={{ padding: '1.5rem', display: 'flex', gap: '1rem', textAlign: 'left', opacity: 0.6, cursor: 'pointer', borderStyle: 'dashed' }}
                  >
                    <div style={{ color: 'var(--accent-primary)' }}><Ruler size={24} /></div>
                    <div>
                      <h4 style={{ textTransform: 'uppercase', fontSize: '0.7rem', fontWeight: 800, color: 'var(--text-dim)', letterSpacing: '1px' }}>Coming Soon</h4>
                      <p style={{ fontWeight: 600 }}>Object Measurement</p>
                    </div>
                  </button>
                  <button
                    onClick={togglePhysics}
                    className="glass-container glow-hover"
                    style={{ padding: '1.5rem', display: 'flex', gap: '1rem', textAlign: 'left', opacity: 0.6, cursor: 'pointer', borderStyle: 'dashed' }}
                  >
                    <div style={{ color: 'var(--accent-primary)' }}><Zap size={24} /></div>
                    <div>
                      <h4 style={{ textTransform: 'uppercase', fontSize: '0.7rem', fontWeight: 800, color: 'var(--text-dim)', letterSpacing: '1px' }}>Coming Soon</h4>
                      <p style={{ fontWeight: 600 }}>Physics Breakdown</p>
                    </div>
                  </button>
                </div>

                <div className="flex-center" style={{ marginTop: '2rem' }}>
                  <button onClick={handleReset} style={{ background: 'transparent', border: 'none', color: 'var(--text-dim)', cursor: 'pointer', fontSize: '0.9rem', fontWeight: 600, textDecoration: 'underline' }}>
                    Generate Another Asset
                  </button>
                </div>
              </div>
            )}
          </div>
        )}
      </main>

      <footer style={{ marginTop: '8rem', padding: '3rem 0', textAlign: 'center', borderTop: '1px solid var(--border-glass)' }}>
        <div className="flex-center gap-medium" style={{ marginBottom: '1.5rem' }}>
          <span style={{ color: 'var(--text-dim)', fontSize: '0.8rem' }}>&copy; 2026 3DGAN Labs</span>
          <span style={{ color: 'var(--text-dim)', fontSize: '0.8rem' }}>&bull;</span>
          <span style={{ color: 'var(--text-dim)', fontSize: '0.8rem' }}>Terms of Service</span>
          <span style={{ color: 'var(--text-dim)', fontSize: '0.8rem' }}>&bull;</span>
          <span style={{ color: 'var(--text-dim)', fontSize: '0.8rem' }}>Research</span>
        </div>
      </footer>
    </div>
  );
}

export default App;
