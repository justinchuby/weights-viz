import react from "@vitejs/plugin-react";
import { defineConfig, type Plugin } from "vite";
import { createHash } from "node:crypto";
import { readFile, readdir, writeFile } from "node:fs/promises";
import { resolve } from "node:path";

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

function injectServiceWorkerAssets(): Plugin {
  return {
    name: "weights-viz-service-worker-assets",
    apply: "build",
    async closeBundle() {
      const outputDirectory = resolve(import.meta.dirname, "dist");
      const assetNames = (await readdir(resolve(outputDirectory, "assets")))
        .sort()
        .map((name) => `./assets/${name}`);
      const buildId = createHash("sha256")
        .update(assetNames.join("\n"))
        .digest("hex")
        .slice(0, 12);
      const serviceWorkerPath = resolve(outputDirectory, "sw.js");
      const serviceWorker = await readFile(serviceWorkerPath, "utf8");
      await writeFile(
        serviceWorkerPath,
        serviceWorker
          .replace("__WEIGHTS_VIZ_CACHE__", `weights-viz-shell-${buildId}`)
          .replace(
            '"__VITE_ASSETS__"',
            assetNames.map((name) => JSON.stringify(name)).join(",\n  ")
          )
      );
    }
  };
}

export default defineConfig({
  plugins: [react(), keepNativeModuleScript(), injectServiceWorkerAssets()],
  base: "./",
  worker: { format: "es" }
});
