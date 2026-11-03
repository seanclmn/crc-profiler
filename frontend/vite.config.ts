import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  // Pre-bundling breaks onnxruntime-web's lookup of its .wasm files in dev.
  optimizeDeps: {
    exclude: ['onnxruntime-web'],
  },
})
