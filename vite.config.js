import { defineConfig } from 'vite';

export default defineConfig({
  server: {
    host: true, // Allow external access for testing on phone
    port: 3000,
    https: true // Required for camera access on mobile
  },
  build: {
    target: 'es2020',
    outDir: 'dist'
  }
});
