import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { fileURLToPath } from 'url'

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      // Quando il codice cerca mediapipe, Vite gli darà il nostro file vuoto
      '@mediapipe/pose': fileURLToPath(new URL('./src/mocks/mediapipe-pose.js', import.meta.url)),
    },
  },
})