import { test, expect } from '@playwright/test';
import path from 'path';
import { fileURLToPath } from 'url';

// ESM-compatible __dirname polyfill (Playwright runs as ESM)
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Test image lives one level above the `frontend/` directory
const TEST_IMAGE = path.resolve(__dirname, '../../test_image.png');

test.describe('3D GAN Full Pipeline E2E', () => {

    test('should complete the full 2D to 3D generation lifecycle', async ({ page }) => {
        // 1. Load the application
        await page.goto('/');
        await expect(page.locator('h1')).toContainText('3D Asset');

        // 2. Upload an image via the hidden <input type="file">
        await page.setInputFiles('input[type="file"]', TEST_IMAGE);

        // 3. Verify transition to progress/status panel
        await expect(page.locator('.status-badge').first()).toBeVisible();

        // 4. Wait for completion (up to 90 s — accounts for GPU inference)
        const completedBadge = page.locator('.status-completed');
        await completedBadge.waitFor({ state: 'visible', timeout: 90_000 });

        // 5. Verify the Three.js canvas is present in the DOM
        const canvas = page.locator('canvas').first();
        await expect(canvas).toBeVisible();

        // 6. Verify the wireframe toggle button is rendered and clickable
        const wireframeBtn = page.locator('button[title="Toggle Wireframe"]');
        await expect(wireframeBtn).toBeVisible();
        await wireframeBtn.click();

        // 7. Verify placeholder slots rendered for Phase 2.4 future features
        await expect(page.locator('text=Measurement Tool')).toBeVisible();
        await expect(page.locator('text=Physics Overlay')).toBeVisible();
    });

    test('should recover gracefully from a failed generation', async ({ page }) => {
        // Mock both status and details to ensure deterministic failure state
        await page.route('**/v1/jobs/*', async route => {
            const url = route.request().url();
            if (url.endsWith('/status')) {
                await route.fulfill({ json: { status: 'FAILED', progress: 45 } });
            } else {
                await route.fulfill({
                    json: {
                        job_id: 'mock-job-id',
                        status: 'FAILED',
                        progress: 45,
                        error_message: 'Inference engine reported a CUDA out-of-memory error',
                        created_at: new Date().toISOString(),
                        updated_at: new Date().toISOString()
                    }
                });
            }
        });

        await page.goto('/');
        await page.setInputFiles('input[type="file"]', TEST_IMAGE);

        // Verify failure state badge appears
        const failedBadge = page.locator('.status-failed');
        await expect(failedBadge).toBeVisible({ timeout: 20_000 });

        // Verify error message is displayed
        await expect(page.locator('text=CUDA out-of-memory error')).toBeVisible();

        // Verify recovery ("Start Over") button is present and clickable
        const retryBtn = page.locator('button:has-text("Start Over")');
        await expect(retryBtn).toBeVisible();
        await retryBtn.click();

        // After clicking reset, we should be back to the upload screen
        await expect(page.locator('text=Drop your image here')).toBeVisible();
    });

});
