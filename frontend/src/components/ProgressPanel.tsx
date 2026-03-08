import React from 'react';
import type { JobDetailResponse } from '../api/types';
import { Loader2, CheckCircle2, AlertCircle, Clock, Cpu, Activity } from 'lucide-react';

interface Props {
    job: JobDetailResponse;
    onReset: () => void;
}

export const ProgressPanel: React.FC<Props> = ({ job, onReset }) => {
    const getStatusConfig = () => {
        switch (job.status) {
            case 'QUEUED':
                return {
                    icon: <Clock className="animate-pulse" size={24} />,
                    color: '#facc15',
                    label: 'In Queue',
                    desc: 'Waiting for an available GPU node...'
                };
            case 'PROCESSING':
                return {
                    icon: <Loader2 className="animate-spin" size={24} />,
                    color: 'var(--accent-primary)',
                    label: 'Processing',
                    desc: 'Our GAN is dreaming up your 3D mesh...'
                };
            case 'COMPLETED':
                return {
                    icon: <CheckCircle2 size={24} />,
                    color: '#22c55e',
                    label: 'Complete',
                    desc: 'Model reconstructed successfully.'
                };
            case 'FAILED':
                return {
                    icon: <AlertCircle size={24} />,
                    color: '#ef4444',
                    label: 'Failed',
                    desc: job.error_message || 'An internal generation error occurred.'
                };
            default:
                return {
                    icon: <Activity size={24} />,
                    color: 'var(--text-dim)',
                    label: job.status,
                    desc: 'Updating state...'
                };
        }
    };

    const config = getStatusConfig();

    return (
        <div className="glass-container" style={{ padding: '2rem' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '2rem' }}>
                <div style={{ display: 'flex', gap: '1.25rem', alignItems: 'center' }}>
                    <div style={{
                        color: config.color,
                        background: `${config.color}15`,
                        padding: '12px',
                        borderRadius: '16px',
                        boxShadow: `0 0 20px -5px ${config.color}40`,
                        display: 'flex'
                    }}>
                        {config.icon}
                    </div>
                    <div style={{ display: 'flex', flexDirection: 'column' }}>
                        <h2 style={{ fontSize: '1.5rem', fontWeight: 800, marginBottom: '0.25rem' }}>{config.label}</h2>
                        <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem' }}>{config.desc}</p>
                    </div>
                </div>

                <div style={{ textAlign: 'right' }}>
                    <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.75rem', color: 'var(--text-dim)', background: 'rgba(255,255,255,0.03)', padding: '4px 8px', borderRadius: '6px' }}>
                        ID: {job.job_id.slice(0, 8)}...
                    </span>
                </div>
            </div>

            <div className="progress-track">
                <div
                    className="progress-bar"
                    style={{ width: `${job.progress}%` }}
                ></div>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '1rem', marginTop: '1rem' }}>
                <div className="flex-col" style={{ gap: '4px', display: 'flex', flexDirection: 'column' }}>
                    <span style={{ fontSize: '0.7rem', color: 'var(--text-dim)', textTransform: 'uppercase', letterSpacing: '1px' }}>Progression</span>
                    <span style={{ fontWeight: 800, fontSize: '1.1rem' }}>{job.progress}%</span>
                </div>
                <div className="flex-col" style={{ gap: '4px', display: 'flex', flexDirection: 'column' }}>
                    <span style={{ fontSize: '0.7rem', color: 'var(--text-dim)', textTransform: 'uppercase', letterSpacing: '1px' }}>Engine</span>
                    <div className="flex-center gap-small" style={{ justifyContent: 'flex-start', display: 'flex', alignItems: 'center', gap: '8px' }}>
                        <Cpu size={14} style={{ color: 'var(--accent-secondary)' }} />
                        <span style={{ fontWeight: 800, fontSize: '1.1rem' }}>RTX 3050</span>
                    </div>
                </div>
                <div className="flex-col" style={{ gap: '4px', display: 'flex', flexDirection: 'column' }}>
                    <span style={{ fontSize: '0.7rem', color: 'var(--text-dim)', textTransform: 'uppercase', letterSpacing: '1px' }}>Latency</span>
                    <span style={{ fontWeight: 800, fontSize: '1.1rem' }}>{job.status === 'PROCESSING' ? '45ms' : '--'}</span>
                </div>
            </div>

            {(job.status === 'FAILED' || job.status === 'EXPIRED') && (
                <button
                    onClick={onReset}
                    className="premium-button"
                    style={{ width: '100%', marginTop: '2rem' }}
                >
                    Try Again
                </button>
            )}
        </div>
    );
};
