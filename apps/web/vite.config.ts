import react from "@vitejs/plugin-react";
import { defineConfig, type Plugin } from "vite";

// Cloudflare Rocket Loader rewrites `type="module"` to a placeholder type and
// re-executes the bundle itself, which delays and can break app startup.
// `data-cfasync="false"` opts the entry script out of that transform.
function keepNativeModuleScript(): Plugin {
  return {
    name: "weights-viz-native-module-script",
    transformIndexHtml: {
      order: "post",
      handler: (html) =>
        html.replace(/<script (?=[^>]*type="module")/g, '<script data-cfasync="false" ')
    }
  };
}

export default defineConfig({
  plugins: [react(), keepNativeModuleScript()],
  base: "./",
  worker: { format: "es" }
});
