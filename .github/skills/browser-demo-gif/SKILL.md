---
name: browser-demo-gif
description: Create polished, deterministic animated GIF demos of web interfaces with Playwright-driven frame capture, a synthetic cursor, explanatory captions, and optimized FFmpeg encoding. Use when adding a UI walkthrough, feature demo, or interaction GIF to documentation.
---

# Browser demo GIF

Create documentation GIFs from deterministic browser screenshots instead of
recording the desktop. This makes every interaction repeatable and gives exact
control over cursor movement, captions, timing, dimensions, and file size.

## Why use a synthetic cursor

Browser screenshot APIs do not capture the operating system cursor. Inject a
cursor-shaped DOM element into the page instead:

- It appears in every screenshot.
- Its path can be eased and reproduced exactly.
- Clicks can have a visible pressed state.
- It does not depend on screen-recording permissions or pointer location.

The cursor and captions are recording-only overlays. Remove them when capture
finishes and never add them to application source.

## Workflow

1. Build and serve the real application.
2. Open a realistic, stable demo state with Playwright.
3. Set a fixed viewport.
4. Inject the cursor and caption overlays.
5. Drive controls with accessible role/name locators.
6. Capture zero-padded PNG frames at each animation step.
7. Inspect representative frames before encoding.
8. Generate an optimized palette and GIF with FFmpeg.
9. Verify dimensions, duration, file size, and README rendering.
10. Delete all temporary frames and browser artifacts.

Prefer a short story with three to five actions. Do not spend GIF time on model
downloads, loading spinners, or setup unless that behavior is the subject of
the demo.

## Prepare the capture

Use a temporary directory outside the final documentation assets:

```sh
mkdir -p .tmp-demo-gif
rm -f .tmp-demo-gif/*.png
```

Choose a viewport that remains readable after downscaling. `1200x760` captured
and then scaled to `1000px` wide works well for a GitHub README.

Wait for remote data, fonts, and animations to settle before taking frame zero.
Use fixed public fixtures or local fixtures so the sequence does not change
between runs.

## Inject recording overlays

Run this in the page before capturing:

```js
await page.evaluate(() => {
  document.getElementById("demo-cursor")?.remove();
  document.getElementById("demo-caption")?.remove();

  const cursor = document.createElement("div");
  cursor.id = "demo-cursor";
  Object.assign(cursor.style, {
    position: "fixed",
    left: "1080px",
    top: "36px",
    width: "18px",
    height: "18px",
    transform: "translate(-2px, -2px)",
    border: "2px solid #1f2328",
    borderRadius: "50%",
    background: "rgba(255, 255, 255, 0.92)",
    boxShadow: "0 1px 5px rgba(0, 0, 0, 0.45)",
    pointerEvents: "none",
    transition: "none",
    zIndex: "2147483647"
  });

  const caption = document.createElement("div");
  caption.id = "demo-caption";
  Object.assign(caption.style, {
    position: "fixed",
    left: "50%",
    bottom: "18px",
    transform: "translateX(-50%)",
    padding: "8px 14px",
    borderRadius: "999px",
    color: "#fff",
    background: "rgba(31, 35, 40, 0.92)",
    boxShadow: "0 2px 10px rgba(0, 0, 0, 0.3)",
    font: '600 13px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
    pointerEvents: "none",
    opacity: "0",
    zIndex: "2147483646"
  });

  document.body.append(cursor, caption);
});
```

Use a circular cursor when the demo should feel neutral and diagrammatic. A CSS
arrow can be used instead, but circles obscure less content and make click
targets easier to see.

## Capture deterministic frames

Create small helpers rather than relying on CSS animation timing:

```js
let frame = 0;
const outputDirectory = "/absolute/path/to/repo/.tmp-demo-gif";

const shot = async () => {
  await page.screenshot({
    path: `${outputDirectory}/${String(frame++).padStart(4, "0")}.png`,
    scale: "css"
  });
};

const hold = async (count) => {
  for (let index = 0; index < count; index += 1) {
    await shot();
  }
};

const setCaption = async (text, visible = true) => {
  await page.evaluate(
    ({ text, visible }) => {
      const caption = document.getElementById("demo-caption");
      caption.textContent = text;
      caption.style.opacity = visible ? "1" : "0";
    },
    { text, visible }
  );
};

const setCursor = async (x, y, pressed = false) => {
  await page.evaluate(
    ({ x, y, pressed }) => {
      const cursor = document.getElementById("demo-cursor");
      cursor.style.left = `${x}px`;
      cursor.style.top = `${y}px`;
      cursor.style.transform =
        `translate(-2px, -2px) scale(${pressed ? 0.72 : 1})`;
      cursor.style.background = pressed
        ? "#0969da"
        : "rgba(255, 255, 255, 0.92)";
    },
    { x, y, pressed }
  );
};
```

Use an absolute output path because a Playwright tool server may run with a
different working directory than the repository.

Move the cursor manually with cubic easing. Capturing each position as a frame
is more deterministic than waiting for a CSS transition:

```js
let cursorX = 1080;
let cursorY = 36;

const moveCursor = async (x, y, frameCount = 7) => {
  const startX = cursorX;
  const startY = cursorY;

  for (let index = 1; index <= frameCount; index += 1) {
    const progress = index / frameCount;
    const eased = 1 - Math.pow(1 - progress, 3);
    await setCursor(
      startX + (x - startX) * eased,
      startY + (y - startY) * eased
    );
    await shot();
  }

  cursorX = x;
  cursorY = y;
};
```

Show a click with one compressed, colored frame:

```js
const clickAt = async (x, y) => {
  await setCursor(x, y, true);
  await shot();
  await page.mouse.click(x, y);
  await setCursor(x, y, false);
  await page.waitForTimeout(200);
  await shot();
};
```

Find targets by semantics, then derive the cursor coordinates from their actual
layout:

```js
const target = page.getByRole("button", { name: /Removed/ });
const box = await target.boundingBox();
if (!box) throw new Error("Demo target is not visible");

const x = box.x + box.width / 2;
const y = box.y + box.height / 2;

await moveCursor(x, y);
await setCaption("Filter tensors removed from the right model");
await clickAt(x, y);
await hold(12);
```

At 10 FPS, `hold(10)` pauses for one second. Keep captions short, describe the
result rather than the mouse motion, and leave enough time to read them.

Wrap recording in `try/finally` when possible. Always remove overlays:

```js
await page.evaluate(() => {
  document.getElementById("demo-cursor")?.remove();
  document.getElementById("demo-caption")?.remove();
});
```

## Encode with FFmpeg

Do not encode directly from screenshots with a generic image converter. Generate
a palette from the complete sequence, then apply it:

```sh
ffmpeg -y -loglevel error \
  -framerate 10 \
  -i .tmp-demo-gif/%04d.png \
  -vf "fps=10,scale=1000:-1:flags=lanczos,\
split[s0][s1];\
[s0]palettegen=max_colors=192:stats_mode=diff[p];\
[s1][p]paletteuse=dither=bayer:bayer_scale=3:diff_mode=rectangle" \
  docs/feature-demo.gif
```

The important options are:

- `palettegen`: builds a palette tailored to the recorded UI.
- `stats_mode=diff`: prioritizes colors in changing regions.
- `diff_mode=rectangle`: optimizes frames around their changed rectangle.
- `bayer_scale=3`: reduces gradients and shadows without excessive noise.
- `scale=1000:-1`: keeps README dimensions predictable.
- `max_colors=192`: usually preserves UI colors while controlling size.

For mostly static interfaces, target 8-12 FPS and less than 1 MiB. Increase
colors before increasing resolution when text or status colors look damaged.

## Validate

Inspect the first frame and one frame after every interaction. Confirm that:

- The caption matches the visible state.
- The fake cursor is centered on the intended control.
- No tooltip, focus ring, or stale selection obscures the feature.
- Text remains readable at the final width.
- The final frame transitions cleanly back to the first frame.

Inspect the encoded artifact:

```sh
ffprobe -v error \
  -show_entries format=duration,size \
  -show_entries stream=width,height,avg_frame_rate \
  -of default=noprint_wrappers=1 \
  docs/feature-demo.gif
```

Add descriptive alt text:

```md
![Model comparison filtering removed and added tensors](docs/feature-demo.gif)
```

Finally remove `.tmp-demo-gif` and any Playwright screenshots, snapshots, or
console logs that are not intended for source control.

## Common failures

- **Cursor is missing:** the real OS cursor is never part of `page.screenshot`;
  verify that the synthetic cursor has a fixed position and high `z-index`.
- **Animation feels robotic:** use eased movement and fewer, longer holds.
- **Caption describes the wrong state:** set it immediately before the action
  and inspect milestone frames.
- **GIF is huge:** reduce FPS, scale width, color count, or idle frames; keep
  `stats_mode=diff` and `diff_mode=rectangle`.
- **UI changes between runs:** avoid coordinate constants for controls; use
  accessible locators and `boundingBox()`.
- **Capture contains loading states:** wait for a stable semantic element rather
  than sleeping for an arbitrary long duration.
- **README looks blurry:** capture larger than the final output and downscale
  once with Lanczos.
