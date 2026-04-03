import { fileURLToPath, URL } from 'node:url'
import { defineConfig, loadEnv } from 'vite'
import vue from '@vitejs/plugin-vue'

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '')
    return {
      plugins: [vue()],
    resolve: {
      alias: {
        '@': fileURLToPath(new URL('./src', import.meta.url))
      }
    },
    server: {
      proxy: {
        '^/api': {
          target: env.VITE_API_URL || 'http://127.0.0.1:5050',
          changeOrigin: true,
          rewrite: (path) => path.replace(/^\/api/, '/api')
        }
      },
      watch: {
        usePolling: true,
        ignored: ['**/node_modules/**', '**/dist/**'],
      },
      host: '0.0.0.0',
    },
    build: {
      sourcemap: false,
      rollupOptions: {
        output: {
          manualChunks(id) {
            if (!id.includes('node_modules')) return undefined
            if (id.includes('echarts')) return 'vendor-echarts'
            if (id.includes('@antv/g6') || id.includes('sigma')) return 'vendor-graph'
            if (id.includes('ant-design-vue')) return 'vendor-antdv'
            if (id.includes('vue') || id.includes('pinia') || id.includes('vue-router')) return 'vendor-vue-core'
            return 'vendor-misc'
          },
        },
      },
    },
  }
})
