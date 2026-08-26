import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [react()],
  base: "./",
  build: {
    outDir: "dist/webview",
    emptyOutDir: false,
    cssCodeSplit: false,
    rollupOptions: {
      input: "src/webview.html",
      output: {
        entryFileNames: "assets/webview.js",
        assetFileNames: "assets/style[extname]"
      }
    }
  }
});
