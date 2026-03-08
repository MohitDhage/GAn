import { defineConfig, devices } from '@playwright/test';

const isCI = Boolean(process.env['CI']);

/**
 * Playwright configuration for Phase 2.4 E2E testing.
 * @see https://playwright.dev/docs/test-configuration
 */
export default defineConfig({
    testDir: './e2e',
    tsconfig: './tsconfig.playwright.json',
    fullyParallel: false,
    forbidOnly: isCI,
    retries: isCI ? 2 : 0,
    workers: 1, // Serial — only one GPU task at a time
    timeout: 120_000, // 2 min per test (GPU inference can be ~10 s + browser startup)
    expect: {
        timeout: 30_000,
    },
    reporter: [['html', { open: 'never' }]],
    use: {
        baseURL: 'http://localhost:5173',
        trace: 'on-first-retry',
        screenshot: 'only-on-failure',
    },
    projects: [
        {
            name: 'chromium',
            use: { ...devices['Desktop Chrome'] },
        },
    ],
    /* Start the Vite dev server before running tests */
    webServer: {
        command: 'npm run dev',
        url: 'http://localhost:5173',
        reuseExistingServer: !isCI,
        timeout: 120_000,
    },
});
