import { defineConfig } from 'vite'

export default defineConfig({
  server: {
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8000', // Uvicorn nel container
        changeOrigin: true,
        secure: false,
        // usa la rewrite SOLO se nel backend le route NON hanno /api
        // rewrite: p => p.replace(/^\/api/, ''),
      },
    },
  },
})
