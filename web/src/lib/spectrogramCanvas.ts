/** Deterministic pseudo-spectrogram for mock cards (no audio file). */
export function drawMockSpectrogram(
  canvas: HTMLCanvasElement,
  seed: string,
  width = 320,
  height = 96
): void {
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  canvas.width = width * dpr;
  canvas.height = height * dpr;
  canvas.style.width = `${width}px`;
  canvas.style.height = `${height}px`;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

  let h = 0;
  for (let i = 0; i < seed.length; i++) h = (h * 31 + seed.charCodeAt(i)) >>> 0;

  const cols = 48;
  const rows = 24;
  const cw = width / cols;
  const rh = height / rows;

  for (let x = 0; x < cols; x++) {
    for (let y = 0; y < rows; y++) {
      const n = Math.sin((h + x * 17 + y * 13) * 0.01) * 0.5 + 0.5;
      const v = Math.pow(n, 1.8);
      const g = 30 + v * 180;
      ctx.fillStyle = `rgb(${g * 0.4},${g * 0.85},${g * 0.55})`;
      ctx.fillRect(x * cw, height - (y + 1) * rh, cw + 0.5, rh + 0.5);
    }
    h = (h * 1664525 + 1013904223) >>> 0;
  }
}

function hann(n: number, N: number): number {
  return 0.5 * (1 - Math.cos((2 * Math.PI * n) / (N - 1)));
}

function binMagnitudes(window: Float32Array, bins: number): Float32Array {
  const N = window.length;
  const out = new Float32Array(bins);
  for (let k = 0; k < bins; k++) {
    let re = 0;
    let im = 0;
    for (let n = 0; n < N; n++) {
      const angle = (-2 * Math.PI * k * n) / N;
      re += window[n] * Math.cos(angle);
      im += window[n] * Math.sin(angle);
    }
    out[k] = Math.sqrt(re * re + im * im);
  }
  return out;
}

/** Decode file and draw a simple spectrogram (STFT magnitude). */
export async function drawSpectrogramFromFile(
  canvas: HTMLCanvasElement,
  file: File,
  width = 320,
  height = 96
): Promise<void> {
  const ctx = canvas.getContext("2d");
  if (!ctx) return;

  const arrayBuf = await file.arrayBuffer();
  const ctxAudio = new AudioContext();
  const buffer = await ctxAudio.decodeAudioData(arrayBuf.slice(0));
  await ctxAudio.close();

  const data = buffer.getChannelData(0);
  const winSize = 1024;
  const hop = 256;
  const bins = 64;
  const cols = Math.min(128, Math.floor((data.length - winSize) / hop) + 1);

  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  canvas.width = width * dpr;
  canvas.height = height * dpr;
  canvas.style.width = `${width}px`;
  canvas.style.height = `${height}px`;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

  const mag = new Float32Array(cols * bins);
  let max = 1e-8;
  for (let c = 0; c < cols; c++) {
    const start = c * hop;
    const w = new Float32Array(winSize);
    for (let i = 0; i < winSize && start + i < data.length; i++) {
      w[i] = data[start + i] * hann(i, winSize);
    }
    const m = binMagnitudes(w, bins);
    for (let b = 0; b < bins; b++) {
      const v = Math.log(1 + m[b]);
      mag[c * bins + b] = v;
      if (v > max) max = v;
    }
  }

  const colW = width / cols;
  const rowH = height / bins;
  for (let c = 0; c < cols; c++) {
    for (let b = 0; b < bins; b++) {
      const v = mag[c * bins + b] / max;
      const g = 20 + v * 220;
      ctx.fillStyle = `rgb(${g * 0.35},${g * 0.75},${g * 0.55})`;
      ctx.fillRect(c * colW, height - (b + 1) * rowH, colW + 0.5, rowH + 0.5);
    }
  }
}
