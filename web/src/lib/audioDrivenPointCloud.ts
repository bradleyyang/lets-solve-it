export interface AudioDrivenPoint {
  index: number;
  emissionTime: number;
  amplitude: number;
  freqHz: number;
}

const TARGET_FPS = 60;
const MIN_FREQ_HZ = 2000;
const MAX_FREQ_HZ = 8000;

function hashStringToSeed(str: string): number {
  let h = 1779033703 ^ str.length;
  for (let i = 0; i < str.length; i++) {
    h = Math.imul(h ^ str.charCodeAt(i), 3432918353);
    h = (h << 13) | (h >>> 19);
  }
  return h >>> 0;
}

function mulberry32(seed: number): () => number {
  return function () {
    let t = (seed += 0x6d2b79f5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function hann(n: number, N: number): number {
  if (N <= 1) return 1;
  return 0.5 * (1 - Math.cos((2 * Math.PI * n) / (N - 1)));
}

function analyzeFrame(
  frame: Float32Array,
  sr: number,
  bins: number
): { amplitude: number; freqHz: number } {
  let sumSq = 0;
  for (let i = 0; i < frame.length; i++) {
    const v = frame[i]!;
    sumSq += v * v;
  }
  const amplitude = Math.min(1, Math.sqrt(sumSq / Math.max(1, frame.length)) * 3.1);

  const N = frame.length;
  let bestEnergy = -1;
  let bestFreq = MIN_FREQ_HZ;
  for (let b = 0; b < bins; b++) {
    const ratio = bins <= 1 ? 0 : b / (bins - 1);
    const freq = MIN_FREQ_HZ + ratio * (MAX_FREQ_HZ - MIN_FREQ_HZ);
    let re = 0;
    let im = 0;
    for (let n = 0; n < N; n++) {
      const angle = (-2 * Math.PI * freq * n) / sr;
      const x = frame[n]!;
      re += x * Math.cos(angle);
      im += x * Math.sin(angle);
    }
    const energy = re * re + im * im;
    if (energy > bestEnergy) {
      bestEnergy = energy;
      bestFreq = freq;
    }
  }

  return { amplitude, freqHz: bestFreq };
}

function frameAudioData(
  data: Float32Array,
  sr: number,
  opts?: { fps?: number; maxSeconds?: number; frameSize?: number }
): AudioDrivenPoint[] {
  const fps = opts?.fps ?? TARGET_FPS;
  const maxSeconds = opts?.maxSeconds ?? 12;
  const frameSize = opts?.frameSize ?? 256;
  const bins = 56;

  const stride = Math.max(1, Math.floor(sr / fps));
  const maxFrames = Math.max(24, Math.floor(maxSeconds * fps));
  const frameCount = Math.min(maxFrames, Math.max(1, Math.floor(data.length / stride)));

  const out: AudioDrivenPoint[] = [];
  for (let i = 0; i < frameCount; i++) {
    const center = i * stride;
    const start = center - Math.floor(frameSize / 2);
    const frame = new Float32Array(frameSize);
    for (let n = 0; n < frameSize; n++) {
      const srcIdx = start + n;
      const sample = srcIdx >= 0 && srcIdx < data.length ? data[srcIdx]! : 0;
      frame[n] = sample * hann(n, frameSize);
    }
    const { amplitude, freqHz } = analyzeFrame(frame, sr, bins);
    out.push({
      index: i,
      emissionTime: i / fps,
      amplitude,
      freqHz,
    });
  }
  return out;
}

function makeSyntheticAudio(seed: string): { samples: Float32Array; sampleRate: number } {
  const rng = mulberry32(hashStringToSeed(seed));
  const sampleRate = 48_000;
  const seconds = 8;
  const N = Math.floor(sampleRate * seconds);
  const out = new Float32Array(N);

  /** Short separated bursts so chirp detection matches “silence → chirp → silence”. */
  const bursts: { start: number; dur: number; f0: number; f1: number }[] = [
    { start: 0.85, dur: 0.26, f0: 2800, f1: 5200 },
    { start: 2.35, dur: 0.2, f0: 3400, f1: 6800 },
    { start: 4.1, dur: 0.3, f0: 2200, f1: 5600 },
  ];

  for (let i = 0; i < N; i++) {
    const t = i / sampleRate;
    let s = 0;
    for (const b of bursts) {
      if (t < b.start || t > b.start + b.dur) continue;
      const u = (t - b.start) / b.dur;
      const env = Math.sin(Math.PI * u) ** 1.6;
      const f = b.f0 + (b.f1 - b.f0) * (u + 0.08 * Math.sin(u * Math.PI * 6));
      const ph = (2 * Math.PI * f * (t - b.start)) / 1;
      s += Math.sin(ph) * env * 0.92;
      s += Math.sin(ph * 2.1 + rng() * 0.4) * env * 0.18;
    }
    out[i] = s + (rng() - 0.5) * 0.012;
  }

  return { samples: out, sampleRate };
}

export async function buildAudioDrivenPoints(
  seed: string,
  file?: File | null
): Promise<AudioDrivenPoint[]> {
  if (file) {
    try {
      const arrayBuf = await file.arrayBuffer();
      const ctx = new AudioContext();
      const decoded = await ctx.decodeAudioData(arrayBuf.slice(0));
      await ctx.close();
      const mono = decoded.getChannelData(0);
      const framed = frameAudioData(mono, decoded.sampleRate);
      if (framed.length > 0) return framed;
    } catch {
      /* fallback to synthetic stream */
    }
  }

  const synthetic = makeSyntheticAudio(seed);
  return frameAudioData(synthetic.samples, synthetic.sampleRate);
}

export function freqToColor(freqHz: number): { r: number; g: number; b: number } {
  const t = Math.max(0, Math.min(1, (freqHz - MIN_FREQ_HZ) / (MAX_FREQ_HZ - MIN_FREQ_HZ)));
  type ColorStop = [number, [number, number, number]];
  const stops = [
    [0.00, [58, 78, 255]],
    [0.18, [186, 56, 255]],
    [0.36, [255, 48, 122]],
    [0.56, [255, 118, 46]],
    [0.72, [255, 234, 64]],
    [0.86, [114, 255, 92]],
    [1.00, [170, 255, 255]],
  ] as ColorStop[];

  let a: ColorStop = stops[0]!;
  let b: ColorStop = stops[stops.length - 1]!;
  for (let i = 0; i < stops.length - 1; i++) {
    const left = stops[i]!;
    const right = stops[i + 1]!;
    if (t >= left[0] && t <= right[0]) {
      a = left;
      b = right;
      break;
    }
  }
  const local = (t - a[0]) / Math.max(1e-6, b[0] - a[0]);
  const c = [0, 1, 2].map((k) => Math.round(a[1][k]! + (b[1][k]! - a[1][k]!) * local));
  return { r: c[0]! / 255, g: c[1]! / 255, b: c[2]! / 255 };
}

/**
 * Frame index chains for each detected “chirp” (contiguous energy above threshold, split by silence gaps).
 * Used to drive sequential square lighting along the temporal path.
 */
export function extractChirpChains(amplitudes: Float32Array, fps: number): number[][] {
  const n = amplitudes.length;
  if (n < 10) {
    return [Array.from({ length: n }, (_, i) => i)];
  }

  const sorted = Array.from(amplitudes).sort((a, b) => a - b);
  const q25 = sorted[Math.floor(n * 0.25)]!;
  const q45 = sorted[Math.floor(n * 0.45)]!;
  const thresh = Math.max(q25 * 1.75, q45 * 1.05, 0.035);

  const gapFrames = Math.max(6, Math.round(0.11 * fps));
  const minChirp = Math.max(5, Math.round(0.07 * fps));
  const maxNodes = 48;

  const chains: number[][] = [];
  let i = 0;
  while (i < n) {
    while (i < n && amplitudes[i]! < thresh) i++;
    if (i >= n) break;
    const start = i;
    let below = 0;
    while (i < n) {
      if (amplitudes[i]! >= thresh) {
        below = 0;
        i++;
      } else {
        below++;
        i++;
        if (below >= gapFrames) break;
      }
    }
    const end = i - below - 1;
    if (end - start + 1 >= minChirp) {
      const raw: number[] = [];
      for (let k = start; k <= end; k++) raw.push(k);
      if (raw.length > maxNodes) {
        const stride = Math.ceil(raw.length / maxNodes);
        chains.push(raw.filter((_, idx) => idx % stride === 0));
      } else {
        chains.push(raw);
      }
    }
  }

  if (chains.length > 0) return chains;

  let best = 0;
  for (let j = 1; j < n; j++) {
    if (amplitudes[j]! > amplitudes[best]!) best = j;
  }
  const half = Math.min(18, Math.floor(n / 3));
  const lo = Math.max(0, best - half);
  const hi = Math.min(n - 1, best + half);
  const fallback: number[] = [];
  for (let k = lo; k <= hi; k++) fallback.push(k);
  return [fallback];
}

/**
 * Indices of frames worth emphasizing (local amplitude peaks above an adaptive floor).
 * Non-peak frames stay dim in the 3D view; only peaks get square labels + strong glow.
 */
export function computeHighlightFrameIndices(amplitudes: Float32Array): number[] {
  const n = amplitudes.length;
  if (n === 0) return [];
  if (n < 7) {
    let best = 0;
    for (let i = 1; i < n; i++) {
      if (amplitudes[i]! > amplitudes[best]!) best = i;
    }
    return [best];
  }

  const sorted = Array.from(amplitudes).sort((a, b) => a - b);
  const median = sorted[Math.floor(n * 0.5)]!;
  const p90 = sorted[Math.floor(n * 0.9)]!;
  const p10 = sorted[Math.floor(n * 0.1)]!;
  const thresh = Math.max(
    median + (p90 - median) * 0.42,
    p10 * 5.5 + 0.025,
    0.055
  );

  const peaks: { i: number; v: number }[] = [];
  for (let i = 1; i < n - 1; i++) {
    const v = amplitudes[i]!;
    if (v < thresh) continue;
    const L = amplitudes[i - 1]!;
    const R = amplitudes[i + 1]!;
    if (v < L || v < R) continue;
    if (v === L && v === R) continue;
    peaks.push({ i, v });
  }

  peaks.sort((a, b) => b.v - a.v);
  const minSep = Math.max(4, Math.floor(n / 160));
  const maxCount = Math.max(16, Math.min(96, Math.ceil(n / 18)));
  const picked: number[] = [];
  for (const p of peaks) {
    if (picked.some((j) => Math.abs(j - p.i) < minSep)) continue;
    picked.push(p.i);
    if (picked.length >= maxCount) break;
  }
  picked.sort((a, b) => a - b);

  if (picked.length === 0) {
    let best = 0;
    for (let i = 1; i < n; i++) {
      if (amplitudes[i]! > amplitudes[best]!) best = i;
    }
    return [best];
  }
  return picked;
}

export function buildTemporalEdges(points: AudioDrivenPoint[]): Array<[number, number]> {
  const edges = new Set<string>();
  for (let i = 1; i < points.length; i++) {
    edges.add(`${i - 1}-${i}`);
    if (i > 2) edges.add(`${i - 2}-${i}`);
  }

  const lookback = 18;
  for (let i = 0; i < points.length; i++) {
    let bestJ = -1;
    let bestD = Number.POSITIVE_INFINITY;
    const p = points[i]!;
    const jStart = Math.max(0, i - lookback);
    for (let j = jStart; j < i; j++) {
      const q = points[j]!;
      const dt = p.emissionTime - q.emissionTime;
      const da = p.amplitude - q.amplitude;
      const df = (p.freqHz - q.freqHz) / 2000;
      const d = dt * dt * 0.3 + da * da * 2 + df * df;
      if (d < bestD) {
        bestD = d;
        bestJ = j;
      }
    }
    if (bestJ >= 0) {
      const lo = Math.min(bestJ, i);
      const hi = Math.max(bestJ, i);
      edges.add(`${lo}-${hi}`);
    }
  }

  return [...edges].map((key) => {
    const [a, b] = key.split("-").map(Number) as [number, number];
    return [a, b];
  });
}
