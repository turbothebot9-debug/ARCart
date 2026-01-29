import { defineConfig } from 'vite';

export default defineConfig({
  server: {
    host: true,
    port: 3000
    // Note: HTTPS disabled for local dev
    // localhost is a secure context, so camera works without HTTPS
    // For phone testing, you'll need to set up proper certs
  },
  build: {
    target: 'es2020',
    outDir: 'dist'
  }
});
