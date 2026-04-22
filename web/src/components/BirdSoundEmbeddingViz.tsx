import { useEffect, useRef } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { EffectComposer } from "three/examples/jsm/postprocessing/EffectComposer.js";
import { RenderPass } from "three/examples/jsm/postprocessing/RenderPass.js";
import { UnrealBloomPass } from "three/examples/jsm/postprocessing/UnrealBloomPass.js";
import { CSS2DObject, CSS2DRenderer } from "three/examples/jsm/renderers/CSS2DRenderer.js";
import {
  buildAudioDrivenPoints,
  enrichChirpRgb,
  extractChirpChains,
  freqToColor,
} from "@/lib/audioDrivenPointCloud";

interface BirdSoundEmbeddingVizProps {
  /** Drives deterministic fallback when no real audio file is available. */
  seed: string;
  audioFile?: File | null;
}

type NarrativePhase = "idle" | "lighting" | "hold" | "fade" | "gap";

const FPS = 60;
const IDLE_FIRST_MS = 720;
const IDLE_LOOP_MS = 520;
const NODE_LIGHT_MS = 32;
const HOLD_MS = 420;
const FADE_MS = 1550;
const GAP_MS = 480;
const GHOST_GREY_R = 0.004;
const GHOST_GREY_G = 0.005;
const GHOST_GREY_B = 0.006;

/**
 * WebGL 3D view from frame-level audio: starts empty, then each detected chirp
 * lights a temporal chain of squares in order, holds, fades out, and gaps before the next.
 */
export function BirdSoundEmbeddingViz({ seed, audioFile }: BirdSoundEmbeddingVizProps) {
  const mountRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const mount = mountRef.current;
    if (!mount) return;

    let alive = true;
    let raf = 0;
    let resizeObserver: ResizeObserver | null = null;
    let renderer: THREE.WebGLRenderer | null = null;
    let labelRenderer: CSS2DRenderer | null = null;
    let controls: OrbitControls | null = null;
    let pointsGeometry: THREE.BufferGeometry | null = null;
    let pointsMaterial: THREE.PointsMaterial | null = null;
    let lineGeometry: THREE.BufferGeometry | null = null;
    let lineMaterial: THREE.LineBasicMaterial | null = null;
    let composer: EffectComposer | null = null;
    let bloomPass: UnrealBloomPass | null = null;
    let labelEls: HTMLDivElement[] = [];
    let labels: CSS2DObject[] = [];

    const setup = async () => {
      const points = await buildAudioDrivenPoints(seed, audioFile);
      if (!alive || points.length === 0) return;

      const scene = new THREE.Scene();
      scene.background = new THREE.Color(0x050608);
      scene.fog = new THREE.Fog(0x050608, 4.8, 18);

      const camera = new THREE.PerspectiveCamera(46, 1, 0.08, 90);
      camera.position.set(2.6, 1.6, 3.1);

      renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
      renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
      renderer.outputColorSpace = THREE.SRGBColorSpace;
      renderer.toneMapping = THREE.ACESFilmicToneMapping;
      renderer.toneMappingExposure = 0.92;

      labelRenderer = new CSS2DRenderer();
      labelRenderer.domElement.className = "embedding-viz__labels";
      labelRenderer.domElement.style.position = "absolute";
      labelRenderer.domElement.style.inset = "0";
      labelRenderer.domElement.style.pointerEvents = "none";

      mount.appendChild(renderer.domElement);
      mount.appendChild(labelRenderer.domElement);

      controls = new OrbitControls(camera, renderer.domElement);
      controls.enableDamping = true;
      controls.dampingFactor = 0.07;
      controls.autoRotate = true;
      controls.autoRotateSpeed = 0.28;
      controls.target.set(0, 0.05, 0);
      controls.minDistance = 1.2;
      controls.maxDistance = 8;

      const grid = new THREE.GridHelper(15, 60, 0x1c2638, 0x121820);
      grid.position.y = -1.6;
      grid.visible = false;
      scene.add(grid);

      const basePos = new Float32Array(points.length * 3);
      const positions = new Float32Array(points.length * 3);
      const colors = new Float32Array(points.length * 3);
      const amplitudes = new Float32Array(points.length);
      const emissions = new Float32Array(points.length);
      const freqs = new Float32Array(points.length);
      const totalDur = points[points.length - 1]?.emissionTime ?? 1;

      for (let i = 0; i < points.length; i++) {
        const p = points[i]!;
        const t = p.emissionTime;
        const tNorm = t / Math.max(0.001, totalDur);
        const amp = p.amplitude;
        const normF = Math.max(0, Math.min(1, (p.freqHz - 2000) / 6000));
        const x = (tNorm - 0.5) * 4.2 + Math.sin(t * 0.8 + normF * 2.1) * 0.42;
        const y = (normF - 0.5) * 1.65 + Math.sin(t * 2.3 + amp * 2.4) * (0.26 + amp * 0.3);
        const z =
          Math.cos(t * 1.6 + normF * 3.2) * (0.55 + amp * 0.9) +
          Math.sin(t * 0.62 + amp * 4.4) * 0.34;
        basePos[i * 3] = x;
        basePos[i * 3 + 1] = y;
        basePos[i * 3 + 2] = z;
        positions[i * 3] = x;
        positions[i * 3 + 1] = y;
        positions[i * 3 + 2] = z;

        amplitudes[i] = amp;
        emissions[i] = t;
        freqs[i] = p.freqHz;
      }

      const chains = extractChirpChains(amplitudes, FPS);
      let maxChain = 0;
      let maxSeg = 0;
      for (const c of chains) {
        maxChain = Math.max(maxChain, c.length);
        maxSeg = Math.max(maxSeg, Math.max(0, c.length - 1));
      }
      const allocSegs = Math.max(1, maxSeg);

      for (let i = 0; i < points.length; i++) {
        colors[i * 3] = 0;
        colors[i * 3 + 1] = 0;
        colors[i * 3 + 2] = 0;
      }

      pointsGeometry = new THREE.BufferGeometry();
      pointsGeometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
      pointsGeometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));
      pointsMaterial = new THREE.PointsMaterial({
        size: 0.19,
        vertexColors: true,
        sizeAttenuation: true,
        transparent: true,
        opacity: 1,
        depthWrite: false,
        blending: THREE.AdditiveBlending,
      });
      const pointCloud = new THREE.Points(pointsGeometry, pointsMaterial);
      scene.add(pointCloud);

      const linePositions = new Float32Array(allocSegs * 6);
      const lineColors = new Float32Array(allocSegs * 6);
      lineGeometry = new THREE.BufferGeometry();
      lineGeometry.setAttribute("position", new THREE.BufferAttribute(linePositions, 3));
      lineGeometry.setAttribute("color", new THREE.BufferAttribute(lineColors, 3));
      lineGeometry.setDrawRange(0, 0);
      lineMaterial = new THREE.LineBasicMaterial({
        transparent: true,
        opacity: 0.55,
        vertexColors: true,
        blending: THREE.AdditiveBlending,
      });
      const lineSegments = new THREE.LineSegments(lineGeometry, lineMaterial);
      scene.add(lineSegments);

      const labelRoot = new THREE.Group();
      for (let j = 0; j < maxChain; j++) {
        const el = document.createElement("div");
        el.className = "embedding-viz__label";
        const ampRow = document.createElement("div");
        ampRow.className = "embedding-viz__label__amp";
        ampRow.textContent = "—";
        const timeRow = document.createElement("div");
        timeRow.className = "embedding-viz__label__t";
        timeRow.textContent = "—";
        el.appendChild(ampRow);
        el.appendChild(timeRow);
        el.style.opacity = "0";
        labelEls.push(el);
        const obj = new CSS2DObject(el);
        obj.position.set(0, 0, 0);
        labels.push(obj);
        labelRoot.add(obj);
      }
      scene.add(labelRoot);

      scene.add(new THREE.AmbientLight(0xffffff, 0.32));

      const dpr = () => Math.min(window.devicePixelRatio, 2);
      const bloomResolution = () => {
        const w = Math.max(320, mount.clientWidth);
        const h = Math.max(420, mount.clientHeight);
        const p = dpr();
        return new THREE.Vector2(Math.max(256, Math.floor(w * p)), Math.max(256, Math.floor(h * p)));
      };
      composer = new EffectComposer(renderer);
      const renderScene = new RenderPass(scene, camera);
      bloomPass = new UnrealBloomPass(bloomResolution(), 1.02, 0.74, 0.048);
      composer.addPass(renderScene);
      composer.addPass(bloomPass);

      const setSize = () => {
        if (!renderer || !labelRenderer || !composer) return;
        const w = Math.max(320, mount.clientWidth);
        const h = Math.max(420, mount.clientHeight);
        const p = dpr();
        camera.aspect = w / h;
        camera.updateProjectionMatrix();
        renderer.setPixelRatio(p);
        renderer.setSize(w, h);
        composer.setPixelRatio(p);
        composer.setSize(w, h);
        labelRenderer.setSize(w, h);
      };
      setSize();
      resizeObserver = new ResizeObserver(() => setSize());
      resizeObserver.observe(mount);

      const narrative: {
        phase: NarrativePhase;
        phaseStart: number;
        chirpIdx: number;
        loopCount: number;
      } = {
        phase: "idle",
        phaseStart: performance.now(),
        chirpIdx: 0,
        loopCount: 0,
      };

      const advance = (phase: NarrativePhase, now: number) => {
        narrative.phase = phase;
        narrative.phaseStart = now;
      };

      const animate = () => {
        if (
          !alive ||
          !renderer ||
          !labelRenderer ||
          !composer ||
          !controls ||
          !pointsGeometry ||
          !lineGeometry ||
          !lineMaterial
        )
          return;
        raf = requestAnimationFrame(animate);
        const now = performance.now();
        const dt = now - narrative.phaseStart;
        const chain = chains[narrative.chirpIdx % chains.length]!;

        let litCount = 0;
        let masterFade = 0;

        switch (narrative.phase) {
          case "idle": {
            const needMs = narrative.loopCount === 0 ? IDLE_FIRST_MS : IDLE_LOOP_MS;
            masterFade = 0;
            if (dt >= needMs) advance("lighting", now);
            break;
          }
          case "lighting": {
            masterFade = 1;
            litCount = Math.min(chain.length, Math.floor(dt / NODE_LIGHT_MS) + 1);
            if (dt >= chain.length * NODE_LIGHT_MS) advance("hold", now);
            break;
          }
          case "hold": {
            masterFade = 1;
            litCount = chain.length;
            if (dt >= HOLD_MS) advance("fade", now);
            break;
          }
          case "fade": {
            masterFade = Math.max(0, 1 - dt / FADE_MS);
            litCount = chain.length;
            if (dt >= FADE_MS) advance("gap", now);
            break;
          }
          case "gap": {
            masterFade = 0;
            litCount = 0;
            if (dt >= GAP_MS) {
              const next = (narrative.chirpIdx + 1) % chains.length;
              narrative.chirpIdx = next;
              if (next === 0) narrative.loopCount++;
              advance("idle", now);
            }
            break;
          }
          default:
            break;
        }

        const posAttr = pointsGeometry.getAttribute("position") as THREE.BufferAttribute;
        const colorAttr = pointsGeometry.getAttribute("color") as THREE.BufferAttribute;
        const lineAttr = lineGeometry.getAttribute("position") as THREE.BufferAttribute;
        const lineColorAttr = lineGeometry.getAttribute("color") as THREE.BufferAttribute;

        const litSet = new Set<number>();
        for (let k = 0; k < litCount; k++) litSet.add(chain[k]!);

        const breathe = Math.sin(now * 0.0022) * 0.04;

        for (let i = 0; i < points.length; i++) {
          const bx = basePos[i * 3]!;
          const by = basePos[i * 3 + 1]!;
          const bz = basePos[i * 3 + 2]!;
          if (litSet.has(i)) {
            const amp = amplitudes[i]!;
            const wob = breathe * (0.5 + amp);
            posAttr.setXYZ(i, bx + wob, by + wob * 0.6, bz - wob * 0.4);
          } else {
            posAttr.setXYZ(i, bx, by, bz);
          }

          const c = enrichChirpRgb(freqToColor(freqs[i]!));
          const amp = amplitudes[i]!;
          const isLit = litSet.has(i);
          if (isLit) {
            const bright = 1.45 + amp * 1.85;
            const activeR = c.r * bright;
            const activeG = c.g * bright;
            const activeB = c.b * bright;
            const mix = Math.max(0, Math.min(1, masterFade));
            colorAttr.setXYZ(
              i,
              GHOST_GREY_R * (1 - mix) + activeR * mix,
              GHOST_GREY_G * (1 - mix) + activeG * mix,
              GHOST_GREY_B * (1 - mix) + activeB * mix
            );
          } else {
            colorAttr.setXYZ(i, GHOST_GREY_R, GHOST_GREY_G, GHOST_GREY_B);
          }
        }
        posAttr.needsUpdate = true;
        colorAttr.needsUpdate = true;

        const segCount = litCount >= 2 ? litCount - 1 : 0;
        for (let s = 0; s < segCount; s++) {
          const ia = chain[s]!;
          const ib = chain[s + 1]!;
          const oa = ia * 3;
          const ob = ib * 3;
          const v0 = s * 2;
          const v1 = s * 2 + 1;
          lineAttr.setXYZ(
            v0,
            posAttr.array[oa] as number,
            posAttr.array[oa + 1] as number,
            posAttr.array[oa + 2] as number
          );
          lineAttr.setXYZ(
            v1,
            posAttr.array[ob] as number,
            posAttr.array[ob + 1] as number,
            posAttr.array[ob + 2] as number
          );
          const cA = enrichChirpRgb(freqToColor(freqs[ia]!));
          const cB = enrichChirpRgb(freqToColor(freqs[ib]!));
          const e = masterFade * 1.55;
          lineColorAttr.setXYZ(v0, cA.r * e, cA.g * e, cA.b * e);
          lineColorAttr.setXYZ(v1, cB.r * e, cB.g * e, cB.b * e);
        }
        lineGeometry.setDrawRange(0, segCount > 0 ? segCount * 2 : 0);
        lineAttr.needsUpdate = true;
        lineColorAttr.needsUpdate = true;
        lineMaterial.opacity = 0.7 * Math.max(0.1, masterFade);

        for (let j = 0; j < labels.length; j++) {
          const el = labelEls[j]!;
          const obj = labels[j]!;
          if (j < litCount && masterFade > 0.02) {
            const idx = chain[j]!;
            const oa = idx * 3;
            const amp = amplitudes[idx]!;
            const tEm = emissions[idx]!;
            const freq = freqs[idx]!;
            obj.position.set(
              (posAttr.array[oa] as number) + 0.08,
              (posAttr.array[oa + 1] as number) + 0.15,
              posAttr.array[oa + 2] as number
            );
            el.querySelector(".embedding-viz__label__amp")!.textContent = amp.toFixed(4);
            el.querySelector(".embedding-viz__label__t")!.textContent = tEm.toFixed(2);
            el.setAttribute("title", `${(freq / 1000).toFixed(2)} kHz`);
            const vis = masterFade;
            el.style.opacity = String(0.25 + vis * 0.75);
            el.style.setProperty("--label-glow", String(0.4 + vis * 0.8));
            el.style.transform = `scale(${0.92 + vis * 0.12})`;
            el.classList.toggle("embedding-viz__label--young", narrative.phase === "lighting");
          } else {
            const ghost = j < chain.length ? 0.018 : 0.004;
            el.style.opacity = String(ghost);
            el.style.transform = "scale(0.86)";
            el.classList.remove("embedding-viz__label--young");
          }
        }

        controls.update();
        composer.render();
        labelRenderer.render(scene, camera);
      };
      animate();

      const rootTag = mount.parentElement?.querySelector(".embedding-viz-stats") as HTMLElement | null;
      if (rootTag) {
        rootTag.textContent = `${points.length} frames · ${chains.length} chirp${chains.length === 1 ? "" : "s"} · chain up to ${maxChain} nodes`;
      }
    };

    setup();

    return () => {
      alive = false;
      cancelAnimationFrame(raf);
      resizeObserver?.disconnect();
      controls?.dispose();
      bloomPass?.dispose();
      composer?.dispose();
      pointsGeometry?.dispose();
      pointsMaterial?.dispose();
      lineGeometry?.dispose();
      lineMaterial?.dispose();
      renderer?.dispose();
      if (renderer?.domElement.parentNode === mount) {
        mount.removeChild(renderer.domElement);
      }
      if (labelRenderer?.domElement.parentNode === mount) {
        mount.removeChild(labelRenderer.domElement);
      }
    };
  }, [seed, audioFile]);

  return (
    <div
      ref={mountRef}
      className="embedding-viz"
      role="img"
      aria-label="Three-dimensional visualization: each bird chirp lights a chain of markers along time, then the chain fades away."
    />
  );
}
