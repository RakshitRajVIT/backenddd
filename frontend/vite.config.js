import { defineConfig } from 'vite';

// Keep browser requests same-origin during development. Vite forwards /api to
// FastAPI, eliminating fragile CORS behavior across changing local ports.
export default defineConfig({
  server: {
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
    },
  },
});
