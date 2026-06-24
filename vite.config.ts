import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { fileURLToPath } from 'url'

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      // Inganniamo Vite fornendogli il nostro mock
      '@mediapipe/pose': fileURLToPath(new URL('./src/mocks/mediapipe-pose.js', import.meta.url)),
    },
  },
  optimizeDeps: {
    include: [
      '@tensorflow/tfjs',
      '@tensorflow/tfjs-backend-webgl',
      '@tensorflow-models/pose-detection'
    ]
  }
})
