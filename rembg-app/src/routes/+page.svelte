<script lang="ts">
  import { invoke } from "@tauri-apps/api/core";
  import { listen } from "@tauri-apps/api/event";
  import { save } from "@tauri-apps/plugin-dialog";
  import { onMount } from "svelte";

  type Device = "cpu" | "gpu";
  type GpuBackend = "auto" | "directml" | "cuda";

  type ProgressEvent = {
    requestId: number;
    stage: string;
    url?: string | null;
    downloaded?: number | null;
    total?: number | null;
    done?: boolean | null;
    message?: string | null;
  };

  type RemoveOptions = {
    model: string;
    device: Device;
    gpu_backend: GpuBackend;
    mask_threshold: number | null;
    bgcolor: string | null;
    color_key_tolerance: number | null;
    allow_download: boolean;
    include_mask: boolean;
  };

  type RemoveResult = {
    output_png: number[];
    mask_png?: number[] | null;
  };

  let inputFile = $state<File | null>(null);
  let inputUrl = $state<string | null>(null);
  let outputUrl = $state<string | null>(null);
  let currentOutputUrl = $state<string | null>(null);
  let maskUrl = $state<string | null>(null);
  let outputBytes = $state<Uint8Array | null>(null);

  let status = $state<string>("Pick an image to start.");
  let progress = $state<ProgressEvent | null>(null);
  let busy = $state(false);
  let runId = $state(0);
  let dragActive = $state(false);

  let options = $state<RemoveOptions>({
    model: "u2netp",
    device: "cpu",
    gpu_backend: "auto",
    mask_threshold: null,
    bgcolor: null,
    color_key_tolerance: null,
    allow_download: true,
    include_mask: false,
  });

  let snapshots = $state<
    { id: number; url: string; bytes: Uint8Array; label: string }[]
  >([]);

  onMount(() => {
    let unlisten: null | (() => void) = null;
    (async () => {
      unlisten = await listen<ProgressEvent>("rembg:progress", (e) => {
        if (e.payload.requestId !== runId) return;
        progress = e.payload;
        if (progress?.message) status = progress.message;
        if (progress?.stage === "infer") status = "Running model...";
        if (progress?.stage === "encode") status = "Encoding PNG...";
      });
    })();

    // Prevent the webview from navigating away when a file is dropped.
    const prevent = (e: DragEvent) => {
      e.preventDefault();
    };
    window.addEventListener("dragover", prevent);
    window.addEventListener("drop", prevent);

    return () => {
      try {
        unlisten?.();
      } catch {
        // ignore
      }
      window.removeEventListener("dragover", prevent);
      window.removeEventListener("drop", prevent);
    };
  });

  function setInput(file: File | null) {
    inputFile = file;
    if (inputUrl) URL.revokeObjectURL(inputUrl);
    inputUrl = file ? URL.createObjectURL(file) : null;

    // Reset outputs when switching input.
    if (currentOutputUrl) {
      try {
        URL.revokeObjectURL(currentOutputUrl);
      } catch {
        // ignore
      }
    }
    currentOutputUrl = null;
    outputUrl = null;
    outputBytes = null;
    if (maskUrl) {
      try {
        URL.revokeObjectURL(maskUrl);
      } catch {
        // ignore
      }
    }
    maskUrl = null;
    for (const s of snapshots) {
      try {
        URL.revokeObjectURL(s.url);
      } catch {
        // ignore
      }
    }
    snapshots = [];
  }

  function bytesToUrl(bytes: Uint8Array, mime = "image/png") {
    const blob = new Blob([bytes], { type: mime });
    return URL.createObjectURL(blob);
  }

  async function runRemove() {
    if (!inputFile) return;
    const myId = ++runId;
    busy = true;
    status = "Preparing...";
    progress = null;

    try {
      const buf = await inputFile.arrayBuffer();
      const inputBytes = new Uint8Array(buf);

      const res = (await invoke("remove_background", {
        requestId: myId,
        inputBytes,
        options: {
          ...options,
          mask_threshold: options.mask_threshold ?? null,
          bgcolor: options.bgcolor?.trim() ? options.bgcolor.trim() : null,
          color_key_tolerance: options.color_key_tolerance ?? null,
        },
      })) as RemoveResult;

      // If a newer run started, ignore this result.
      if (myId !== runId) return;

      const out = new Uint8Array(res.output_png);
      outputBytes = out;

      // Manage current output URL separately from snapshot URLs to avoid revoking snapshot previews.
      if (currentOutputUrl) URL.revokeObjectURL(currentOutputUrl);
      currentOutputUrl = bytesToUrl(out);
      outputUrl = currentOutputUrl;

      if (res.mask_png) {
        const m = new Uint8Array(res.mask_png);
        if (maskUrl) URL.revokeObjectURL(maskUrl);
        maskUrl = bytesToUrl(m);
      } else if (maskUrl) {
        URL.revokeObjectURL(maskUrl);
        maskUrl = null;
      }

      status = "Ready.";
      busy = false;

      // Snapshot ring buffer.
      const label = `${options.model} - ${options.device}${
        options.device === "gpu" ? `/${options.gpu_backend}` : ""
      } - t=${options.mask_threshold ?? "off"}`;
      const snapUrl = bytesToUrl(out);
      const next = [{ id: Date.now(), url: snapUrl, bytes: out, label }, ...snapshots];
      const trimmed = next.slice(0, 6);
      for (const s of next.slice(6)) {
        try {
          URL.revokeObjectURL(s.url);
        } catch {
          // ignore
        }
      }
      snapshots = trimmed;
    } catch (e) {
      if (myId !== runId) return;
      busy = false;
      status = `${e}`;
    }
  }

  let debounceTimer: number | null = null;
  function scheduleRun() {
    if (!inputFile) return;
    if (debounceTimer) window.clearTimeout(debounceTimer);
    debounceTimer = window.setTimeout(() => {
      void runRemove();
    }, 250);
  }

  async function exportPng() {
    if (!outputBytes) return;
    const path = await save({
      title: "Export PNG",
      defaultPath: "rembg.png",
      filters: [{ name: "PNG image", extensions: ["png"] }],
    });
    if (!path) return;
    await invoke("write_file_bytes", { path, bytes: outputBytes });
    status = `Saved: ${path}`;
  }
</script>

<main class="app">
  <header class="top">
    <div class="brand">
      <div class="mark">R</div>
      <div class="title">
        <div class="name">rembg</div>
        <div class="tag">Rust + Tauri</div>
      </div>
    </div>
    <div class="actions">
      <label class="file">
        <input
          type="file"
          accept="image/*"
          onchange={(e) => {
            const f = (e.currentTarget as HTMLInputElement).files?.[0] ?? null;
            setInput(f);
            scheduleRun();
          }}
        />
        <span>Choose Image</span>
      </label>
      <button class="btn" disabled={!outputBytes} onclick={exportPng}>Export PNG</button>
    </div>
  </header>

  <section class="grid">
    <div class="panel">
      <div class="panelHead">
        <div class="h">Input</div>
        <div class="sub">{inputFile ? inputFile.name : "No file selected"}</div>
      </div>
      <div
        class={"preview " + (dragActive ? "drag" : "")}
        role="button"
        tabindex="0"
        ondragenter={(e) => {
          e.preventDefault();
          dragActive = true;
        }}
        ondragover={(e) => {
          e.preventDefault();
          dragActive = true;
        }}
        ondragleave={(e) => {
          e.preventDefault();
          dragActive = false;
        }}
        ondrop={(e) => {
          e.preventDefault();
          dragActive = false;
          const f = e.dataTransfer?.files?.[0] ?? null;
          if (!f) return;
          if (!f.type.startsWith("image/")) {
            status = "Dropped file is not an image.";
            return;
          }
          setInput(f);
          scheduleRun();
        }}
      >
        {#if inputUrl}
          <img src={inputUrl} alt="input" />
        {:else}
          <div class="empty">Drop an image here, or use Choose Image.</div>
        {/if}
      </div>
    </div>

    <div class="panel">
      <div class="panelHead">
        <div class="h">Output</div>
        <div class="sub">{busy ? "Processing..." : "Transparent PNG"}</div>
      </div>
      <div class="preview checker">
        {#if outputUrl}
          <img src={outputUrl} alt="output" />
        {:else}
          <div class="empty">Run will appear here.</div>
        {/if}
      </div>
    </div>
  </section>

  <section class="controls">
    <div class="row">
      <div class="ctl">
        <div class="lbl">Model</div>
        <select bind:value={options.model} onchange={scheduleRun}>
          <option value="u2netp">u2netp (fast)</option>
          <option value="u2net">u2net (bigger)</option>
          <option value="silueta">silueta (small)</option>
          <option value="u2net_human_seg">u2net_human_seg</option>
          <option value="u2net_cloth_seg">u2net_cloth_seg</option>
          <option value="isnet-general-use">isnet-general-use (quality, slow)</option>
          <option value="isnet-anime">isnet-anime</option>
        </select>
      </div>

      <div class="ctl">
        <div class="lbl">Device</div>
        <select bind:value={options.device} onchange={scheduleRun}>
          <option value="cpu">CPU</option>
          <option value="gpu">GPU</option>
        </select>
      </div>

      <div class="ctl">
        <div class="lbl">GPU Backend</div>
        <select
          bind:value={options.gpu_backend}
          onchange={scheduleRun}
          disabled={options.device !== "gpu"}
        >
          <option value="auto">Auto</option>
          <option value="directml">DirectML</option>
          <option value="cuda">CUDA</option>
        </select>
      </div>

      <div class="ctl">
        <div class="lbl">Mask Threshold</div>
        <input
          type="range"
          min="0"
          max="255"
          value={options.mask_threshold ?? 0}
          oninput={(e) => {
            const v = Number((e.currentTarget as HTMLInputElement).value);
            options.mask_threshold = v === 0 ? null : v;
            scheduleRun();
          }}
        />
        <div class="hint">{options.mask_threshold ?? "off"}</div>
      </div>

      <div class="ctl">
        <div class="lbl">Color Key Tol</div>
        <input
          type="range"
          min="0"
          max="80"
          value={options.color_key_tolerance ?? 0}
          oninput={(e) => {
            const v = Number((e.currentTarget as HTMLInputElement).value);
            options.color_key_tolerance = v === 0 ? null : v;
            scheduleRun();
          }}
        />
        <div class="hint">{options.color_key_tolerance ?? "off"}</div>
      </div>
    </div>

    <div class="row">
      <div class="ctl wide">
        <div class="lbl">Background Color (optional)</div>
        <input
          placeholder="#RRGGBB (leave empty for transparency)"
          value={options.bgcolor ?? ""}
          oninput={(e) => {
            const s = (e.currentTarget as HTMLInputElement).value;
            options.bgcolor = s.trim() ? s : null;
            scheduleRun();
          }}
        />
      </div>
      <label class="check">
        <input
          type="checkbox"
          checked={options.allow_download}
          onchange={(e) => {
            options.allow_download = (e.currentTarget as HTMLInputElement).checked;
            scheduleRun();
          }}
        />
        <span>Allow downloads (runtime/model)</span>
      </label>
      <label class="check">
        <input
          type="checkbox"
          checked={options.include_mask}
          onchange={(e) => {
            options.include_mask = (e.currentTarget as HTMLInputElement).checked;
            scheduleRun();
          }}
        />
        <span>Also keep mask</span>
      </label>
      {#if maskUrl}
        <a class="maskLink" href={maskUrl} target="_blank">Open mask</a>
      {/if}
    </div>

    <div class="status">
      <div class="left">
        <div class="s">{status}</div>
        {#if progress?.downloaded != null}
          <div class="bar">
            <div
              class="fill"
              style={`width: ${
                progress.total ? Math.min(100, (progress.downloaded / progress.total) * 100) : 25
              }%`}
            ></div>
          </div>
        {/if}
      </div>
      <div class="right">
        <button class="btn" disabled={!inputFile || busy} onclick={runRemove}>Run</button>
      </div>
    </div>

    {#if snapshots.length}
      <div class="shots">
        <div class="shotsHead">Snapshots</div>
        <div class="shotsRow">
          {#each snapshots as s (s.id)}
            <button
              class="shot"
              title={s.label}
              onclick={() => {
                outputUrl = s.url;
                outputBytes = s.bytes;
                status = `Selected snapshot: ${s.label}`;
              }}
            >
              <div class="thumb checker">
                <img src={s.url} alt="snapshot" />
              </div>
              <div class="cap">{s.label}</div>
            </button>
          {/each}
        </div>
      </div>
    {/if}
  </section>
</main>

<style>
  :global(:root) {
    color-scheme: light dark;
    --bg: #ffffff;
    --panel: #ffffff;
    --text: #111111;
    --muted: rgba(0, 0, 0, 0.62);
    --border: rgba(0, 0, 0, 0.12);
    --shadow: rgba(0, 0, 0, 0.08);
    --accent: #0b57d0;
    --accent-2: #111111;
    --checker-bg: #ffffff;
    --checker-a: rgba(0, 0, 0, 0.08);
  }

  @media (prefers-color-scheme: dark) {
    :global(:root) {
      /* Dracula-ish */
      --bg: #282a36;
      --panel: #1f202a;
      --text: #f8f8f2;
      --muted: rgba(248, 248, 242, 0.70);
      --border: rgba(248, 248, 242, 0.12);
      --shadow: rgba(0, 0, 0, 0.35);
      --accent: #bd93f9;
      --accent-2: #ff79c6;
      --checker-bg: #1b1c25;
      --checker-a: rgba(248, 248, 242, 0.10);
    }
  }

  :global(html, body) {
    height: 100%;
    margin: 0;
    background: var(--bg);
    color: var(--text);
    font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, Helvetica, Arial,
      "Apple Color Emoji", "Segoe UI Emoji";
  }

  .app {
    min-height: 100%;
    padding: 18px 18px 28px;
    box-sizing: border-box;
    display: flex;
    flex-direction: column;
    gap: 14px;
  }

  .top {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    padding: 12px 14px;
    border-radius: 14px;
    border: 1px solid var(--border);
    background: var(--panel);
    box-shadow: 0 10px 24px var(--shadow);
  }

  .brand {
    display: flex;
    align-items: center;
    gap: 12px;
  }
  .mark {
    width: 36px;
    height: 36px;
    border-radius: 12px;
    display: grid;
    place-items: center;
    font-weight: 900;
    letter-spacing: -0.04em;
    background: var(--accent);
    color: var(--bg);
  }
  .name {
    font-size: 16px;
    font-weight: 800;
    letter-spacing: 0.01em;
  }
  .tag {
    font-size: 12px;
    color: var(--muted);
  }

  .actions {
    display: flex;
    gap: 10px;
    align-items: center;
  }

  .file input {
    display: none;
  }
  .file span,
  .btn {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: 8px;
    padding: 10px 12px;
    border-radius: 12px;
    background: var(--panel);
    border: 1px solid var(--border);
    color: var(--text);
    cursor: pointer;
    transition: transform 120ms ease, background 140ms ease, border-color 140ms ease;
    font-weight: 700;
  }
  .btn:disabled {
    opacity: 0.45;
    cursor: not-allowed;
  }
  .file span:hover,
  .btn:hover:not(:disabled) {
    border-color: color-mix(in srgb, var(--accent) 45%, var(--border));
  }
  .file span:active,
  .btn:active:not(:disabled) {
    transform: translateY(1px);
  }

  .grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 14px;
  }
  @media (max-width: 980px) {
    .grid {
      grid-template-columns: 1fr;
    }
  }

  .panel {
    border-radius: 16px;
    border: 1px solid var(--border);
    background: var(--panel);
    overflow: hidden;
    box-shadow: 0 10px 24px var(--shadow);
  }
  .panelHead {
    padding: 12px 14px;
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    border-bottom: 1px solid var(--border);
  }
  .h {
    font-weight: 900;
    letter-spacing: 0.02em;
  }
  .sub {
    font-size: 12px;
    color: var(--muted);
    max-width: 60%;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .preview {
    height: clamp(260px, 45vh, 520px);
    display: flex;
    align-items: center;
    justify-content: center;
    position: relative;
    padding: 12px;
    box-sizing: border-box;
  }
  .preview img {
    width: 100%;
    height: 100%;
    object-fit: contain;
    display: block;
    pointer-events: none;
  }
  .preview.drag {
    outline: 2px solid var(--accent);
    outline-offset: -2px;
  }
  .empty {
    padding: 16px;
    color: var(--muted);
    font-weight: 700;
    text-align: center;
  }

  .checker {
    background-color: var(--checker-bg);
    background-image: linear-gradient(45deg, var(--checker-a) 25%, transparent 25%),
      linear-gradient(-45deg, var(--checker-a) 25%, transparent 25%),
      linear-gradient(45deg, transparent 75%, var(--checker-a) 75%),
      linear-gradient(-45deg, transparent 75%, var(--checker-a) 75%);
    background-size: 24px 24px;
    background-position: 0 0, 0 12px, 12px -12px, -12px 0px;
  }

  .controls {
    padding: 14px;
    border-radius: 16px;
    border: 1px solid var(--border);
    background: var(--panel);
    box-shadow: 0 10px 24px var(--shadow);
    display: flex;
    flex-direction: column;
    gap: 12px;
  }

  .row {
    display: flex;
    gap: 12px;
    flex-wrap: wrap;
    align-items: flex-end;
  }

  .ctl {
    min-width: 180px;
    flex: 1;
    display: flex;
    flex-direction: column;
    gap: 6px;
  }
  .ctl.wide {
    flex: 2;
    min-width: 260px;
  }
  .lbl {
    font-size: 12px;
    color: var(--muted);
    font-weight: 800;
    letter-spacing: 0.03em;
    text-transform: uppercase;
  }
  select,
  input:not([type]) {
    padding: 10px 10px;
    border-radius: 12px;
    background: var(--bg);
    border: 1px solid var(--border);
    color: var(--text);
    outline: none;
  }
  select:disabled,
  input:disabled {
    opacity: 0.55;
    cursor: not-allowed;
  }
  /* Try to force readable native option lists (Windows WebView can otherwise show white-on-white). */
  option {
    background: var(--bg);
    color: var(--text);
  }
  input[type="range"] {
    width: 100%;
  }
  .hint {
    font-family: "Cascadia Mono", ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    font-size: 12px;
    color: var(--muted);
  }

  .check {
    display: inline-flex;
    gap: 10px;
    align-items: center;
    padding: 10px 12px;
    border-radius: 12px;
    border: 1px solid var(--border);
    background: transparent;
    font-weight: 700;
    cursor: pointer;
  }
  .check input {
    transform: translateY(1px);
  }
  .maskLink {
    align-self: center;
    color: var(--accent);
    text-decoration: none;
    font-weight: 800;
  }

  .status {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 12px;
    padding-top: 2px;
  }
  .s {
    font-weight: 800;
    opacity: 0.9;
  }
  .bar {
    height: 8px;
    border-radius: 999px;
    margin-top: 8px;
    background: color-mix(in srgb, var(--border) 60%, transparent);
    overflow: hidden;
    border: 1px solid var(--border);
  }
  .fill {
    height: 100%;
    background: linear-gradient(90deg, var(--accent) 0%, var(--accent-2) 100%);
  }

  .shots {
    margin-top: 8px;
  }
  .shotsHead {
    font-weight: 900;
    letter-spacing: 0.02em;
    margin-bottom: 8px;
    opacity: 0.9;
  }
  .shotsRow {
    display: flex;
    gap: 10px;
    overflow-x: auto;
    padding-bottom: 6px;
  }
  .shot {
    min-width: 120px;
    text-align: left;
    background: transparent;
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 6px;
    color: inherit;
    cursor: pointer;
  }
  .thumb {
    border-radius: 12px;
    overflow: hidden;
    aspect-ratio: 16 / 11;
    display: grid;
    place-items: center;
    border: 1px solid var(--border);
    margin-bottom: 8px;
  }
  .thumb img {
    max-width: 100%;
    max-height: 100%;
    object-fit: contain;
  }
  .cap {
    font-size: 12px;
    color: var(--muted);
    font-family: "Cascadia Mono", ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
</style>
