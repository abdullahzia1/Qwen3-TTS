import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// In dev, the API runs separately (python -m server, port 8100).
// In production the API serves the built files itself, same origin.
export default defineConfig({
  plugins: [react()],
  server: { proxy: { "/v1": "http://127.0.0.1:8100" } },
});
