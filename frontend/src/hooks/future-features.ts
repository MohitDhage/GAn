import { useState, useCallback } from 'react';

/**
 * Placeholder hook for future measurement requirements.
 * Defines the architectural slot for dimensioning and scaling tools.
 */
export const useMeasurement = () => {
    const [isMeasuring, setIsMeasuring] = useState(false);
    const [measurements, setMeasurements] = useState<{ x: number; y: number; z: number } | null>(null);

    const startMeasurement = useCallback(() => {
        console.log('Measurement tool: UI slot ready, logic pending for Subphase 3.x');
        setIsMeasuring(true);
    }, []);

    const clearMeasurement = useCallback(() => {
        setIsMeasuring(false);
        setMeasurements(null);
    }, []);

    return { isMeasuring, measurements, startMeasurement, clearMeasurement };
};

/**
 * Placeholder hook for future physics property requirements.
 * Defines the architectural slot for center of mass and collision hull generation.
 */
export const usePhysicsOverlay = () => {
    const [showPhysics, setShowPhysics] = useState(false);
    const [physicsProps] = useState<{ volume: number; mass: number } | null>(null);

    const togglePhysics = useCallback(() => {
        console.log('Physics overlay: UI slot ready, logic pending for Subphase 3.x');
        setShowPhysics(prev => !prev);
    }, []);

    return { showPhysics, physicsProps, togglePhysics };
};
