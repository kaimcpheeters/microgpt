import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// Cross-origin isolation: SharedArrayBuffer (used for Pyodide's interrupt
// buffer, which powers the Stop button without rebooting the worker)
// requires the page to be "cross-origin isolated". That in turn needs
// COOP=same-origin and COEP=require-corp or credentialless. We use
// `credentialless` so cross-origin assets (jsdelivr's Pyodide CDN, Google
// Fonts, the upstream GitHub raw dataset) load without each origin
// needing to opt in via CORP headers.
//
// For production: the same headers must be set on the deploy. See
// `public/_headers` (Netlify / Cloudflare Pages syntax).
const ISOLATION_HEADERS = {
  'Cross-Origin-Opener-Policy': 'same-origin',
  'Cross-Origin-Embedder-Policy': 'credentialless',
}

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: { headers: ISOLATION_HEADERS },
  preview: { headers: ISOLATION_HEADERS },
  worker: {
    format: 'es',
  },
})
