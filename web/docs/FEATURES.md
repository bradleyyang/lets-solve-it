# Web application — feature reference

This document describes **every user-visible feature** and the **code that implements it**, as of the current repository. It is aimed at onboarding engineers who will extend or replace the mock layer with real services.

---

## 1. Product scope (what the web app is)

The web client is a **research-style prototype workspace** for exploring bird-audio workflows **without a server**:

- Browse a **fixed mock catalog** of species rows (text search).
- **Upload** an audio file in the browser (decoded with Web Audio API).
- Run **mock** “classification” and “similarity” that ignore audio content but simulate latency.
- **Save** rows to `localStorage`, **compare** two slots, and open a **3D temporal embedding visualization** driven by frame-level analysis (real audio or synthetic fallback).

Nothing here trains models or calls Xeno-canto; it is a **UI and pipeline sketch** aligned with the team’s longer-term audio–text retrieval direction.

---

## 2. Technology stack

| Layer | Choice |
|-------|--------|
| Build | Vite 6 |
| UI | React 18 + TypeScript |
| Routing | React Router 6 |
| 3D | Three.js (WebGLRenderer, OrbitControls, CSS2D labels, EffectComposer + UnrealBloomPass) |
| Styling | Single global `index.css` (design tokens as CSS variables) |

---

## 3. Application entry and routing

**File:** `src/main.tsx`  
Mounts React root.

**File:** `src/App.tsx`  
Wraps the tree in:

- `BrowserRouter`
- `AppPreferencesProvider` — upload + vocab + compare slots (in-memory except vocab).
- `SavedProvider` — saved list (backed by `localStorage`).

**Routes** (all except `*` use `AppShell` layout):

| Path | Page component | Purpose |
|------|----------------|---------|
| `/` | `HomePage` | Workspace overview, upload entry, links into Query |
| `/query` | `QueryPage` | Catalog search, upload, mock classify/similarity, results grid |
| `/saved` | `SavedPage` | Grid of saved `SearchResult` rows |
| `/compare` | `ComparePage` | Two-slot side-by-side comparison |
| `/viz/:id` | `VizPage` | Species context + lazy 3D embedding viz for `id` |
| `*` | `Navigate` → `/` | Unknown paths bounce home |

**Param:** `VizPage` reads `id` from the URL; it must match a `SearchResult.id` in the mock catalog (`getResultById`).

---

## 4. Shell, navigation, and footer

**File:** `src/layout/AppShell.tsx`

- **Brand block:** Title “Bird audio analysis” + subtitle noting client-side prototype / mock catalog.
- **Navigation:** `NavLink`s — Overview (`/`), Query, Saved, Compare. Compare shows a badge `2` when both compare slots are filled (`useAppPreferences().compareSlots`).
- **Main:** `<Outlet />` renders the active page.
- **Footer:** Disclaimer that embeddings and search are simulated in-browser.

Styling for header/nav/footer lives in `src/index.css` (classes prefixed `app-shell__`).

---

## 5. Feature: Overview (Home)

**File:** `src/pages/HomePage.tsx`

| UI block | Behaviour |
|----------|-----------|
| **Instrument strip** | Mono banner stating prototype / mock / no server. |
| **Page header** | “Overview” + paragraph describing the workspace. |
| **Audio intake** | File input (`accept` audio). On change, calls `setUploadedFile` from `AppPreferences`. |
| **Spectrogram preview** | `useSpectrogram(uploadedFile, canvas)` + `drawSpectrogramFromFile` pipeline (see §10). |
| **Continue to Query** | `Link` to `/query` carrying the in-memory upload (same session). |
| **Catalog search** | Link to `/query?source=dataset` — Query page reads this to seed the search field (see §6). |
| **Taxonomic display** | Segmented control toggles `vocabMode` (`common` \| `scientific`) persisted in `localStorage` (`AppPreferences`). |

**Hook:** `src/hooks/useSpectrogram.ts` — redraws when `file` or `canvas` changes; surfaces decode errors (not heavily surfaced in Home UI today).

---

## 6. Feature: Query workspace

**File:** `src/pages/QueryPage.tsx`  
**API:** `src/api/mock.ts`  
**Types:** `src/api/types.ts`

### 6.1 Vocabulary mode

- Same segmented control as Home: `vocabMode` affects **dataset search** string matching (common vs scientific name primary field).
- Persisted key: `lets-solve-it:vocab` (see `AppPreferences.tsx`).

### 6.2 Catalog (dataset) search

- Input: text search; Enter or button triggers `searchDataset(query, vocabMode)`.
- **Mock logic:** substring match on normalized common name, scientific name, or `speciesCode`. Empty query returns full catalog filter pass; if filter is empty, returns first 8 rows (see `mock.ts`).
- Artificial delay ~400 ms to mimic network.
- Results rendered as `ResultCard` grid.

**URL hint:** `?source=dataset` in `useEffect` seeds `query` to “Turdus” or “sparrow” depending on vocab mode.

### 6.3 Upload and mock classification

- File input updates `uploadedFile` and clears prior `classifyHits`.
- **Spectrogram:** second canvas + `useSpectrogram`.
- **Classify button:** `classifyUpload(uploadedFile)` — **ignores file content**; returns fixed chickadee-heavy label list after ~600 ms.
- Renders ordered list “Posterior over labels (mock)” with scores.

### 6.4 Mock similarity search

- **Search similar** button: `searchSimilarToUpload(uploadedFile)` — ignores file; returns first 6 catalog rows with descending fake `similarity` after ~500 ms.

### 6.5 Result set

- Grid of `ResultCard` for current `results` state.
- Link to `/saved` for saved list.

---

## 7. Feature: Saved specimens

**File:** `src/pages/SavedPage.tsx`  
**State:** `src/context/SavedContext.tsx`  
**Persistence:** `src/saved/savedStore.ts`

- Reads `saved` array from context (derived from `loadSaved()`).
- **Storage key:** `lets-solve-it:saved` — JSON array of `SearchResult` objects.
- **Toggle:** implemented on `ResultCard` via `toggle(result)`; dedupes by `result.id`.
- Empty state: short message when no rows saved.

---

## 8. Feature: Paired comparison

**Files:** `src/pages/ComparePage.tsx`, preferences in `AppPreferences.tsx`

- **Slots:** `[string | null, string | null]` storing **result ids** (`SearchResult.id`).
- **Add:** `ResultCard` calls `addToCompare(result.id)`:
  - Fills slot A, then B; duplicates ignored; if both full, replaces slot A with new id (`CompareAddResult` enum).
  - When second slot is filled (`added_second`), card navigates to `/compare`.
- **Resolve rows:** `getResultById` from mock catalog.
- **Clear:** `clearCompare()` resets both slots.
- Empty slots show dashed placeholder copy.

---

## 9. Feature: Result cards

**File:** `src/components/ResultCard.tsx`  
**Drawing:** `src/lib/spectrogramCanvas.ts`

| Element | Description |
|---------|-------------|
| **Image** | `imageUrl` from mock (currently ui-avatars.com placeholder). |
| **Metadata** | Common + scientific name, vocalization, duration, recording id, optional similarity %. |
| **Spectrogram canvas** | If `spectrogramFile` prop set, draws from file; else `drawMockSpectrogram(canvas, result.id)` deterministic fake pattern. |
| **Embedding** | Navigates to `/viz/:id`. |
| **Compare slot** | `addToCompare` (see §8). |
| **Save** | `toggle` saved state; primary style when saved. |

---

## 10. Feature: Spectrogram preview (shared)

**Files:** `src/hooks/useSpectrogram.ts`, `src/lib/spectrogramCanvas.ts`

- **Real file path:** decode audio with Web Audio (`AudioContext.decodeAudioData`), downmix to mono, compute magnitude STFT-like bins, draw greyscale canvas.
- **Mock path:** hash `result.id` into a stable fake pattern for cards without upload.
- **Errors:** Hook sets error string on failure; not all pages surface it in UI (could be improved).

Used on **Home** and **Query** for the upload preview.

---

## 11. Feature: Temporal embedding visualization

**Route:** `/viz/:id`  
**Files:** `src/pages/VizPage.tsx`, `src/components/BirdSoundEmbeddingViz.tsx`, `src/lib/audioDrivenPointCloud.ts`

### 11.1 Page chrome

- Breadcrumb back to Query.
- Header explains frame-wise analysis at ~60 Hz (conceptually).
- **Stack:** Dark “stage” panel with stats line + WebGL canvas; below, species **footer card** (image + names + recording meta).
- **Data guide:** Static frequency axis legend + short methodology blurb.

### 11.2 Lazy loading

`BirdSoundEmbeddingViz` is `React.lazy` + `Suspense` so Three.js loads only when visiting viz.

### 11.3 Audio → points pipeline (`audioDrivenPointCloud.ts`)

- **`buildAudioDrivenPoints(seed, file?)`:** If `file` provided, decode to mono PCM, else **`makeSyntheticAudio(seed)`** (short separated chirp-like bursts for demo).
- **Framing:** `frameAudioData` — sliding Hann-windowed chunks at **TARGET_FPS 60**, per-frame **amplitude** (RMS) and **dominant frequency** via coarse bank of complex sinusoids (not a full FFT library).
- **`extractChirpChains(amplitudes, fps)`:** Segments “chirps” from amplitude envelope for narrative timing.
- **`freqToColor` / `enrichChirpRgb`:** Jewel-tone ramp + saturation boost for lit points.
- **Exports** also include `computeHighlightFrameIndices` (legacy / unused by current viz narrative).

### 11.4 Three.js scene (`BirdSoundEmbeddingViz.tsx`)

- **Scene:** dark background, fog optional, grid hidden for cleaner idle state.
- **Points:** `BufferGeometry` + `PointsMaterial` (vertex colors, **additive** blending).
- **Lines:** `LineSegments` between consecutive lit chain indices during a chirp.
- **Labels:** `CSS2DRenderer` + `CSS2DObject` divs (mono text beside points, not boxed).
- **Narrative state machine:** `idle` → `lighting` (sequential nodes) → `hold` → `fade` (blend lit color toward dark grey) → `gap` → next chirp / loop.
- **Post-processing:** `EffectComposer` + `RenderPass` + `UnrealBloomPass` for bloom; `ACESFilmicToneMapping` on renderer.
- **Stats line:** DOM `.embedding-viz-stats` updated with frame/chirp counts (sibling of viz in `VizPage` structure — actually updated from viz via `mount.parentElement?.querySelector`).

### 11.5 Viz + upload coupling

`VizPage` passes `audioFile={uploadedFile}` from preferences. If user did not upload on Query, viz uses **synthetic** audio but still keys deterministic layout off `seed` built from `result.id` + `recordingId`.

---

## 12. Type definitions (contract for UI and future API)

**File:** `src/api/types.ts`

- `VocabMode` — used throughout search UI.
- `QuerySource` — exported for future API/source toggles; **not referenced** by page components today.
- `ClassificationHit`, `SearchResult` — shape of mock rows and classifier lines.

Any backend should aim to preserve these fields or provide adapters before the UI is rewritten.

---

## 13. Styling and design language

**File:** `src/index.css`

- CSS variables for **paper-like neutrals**, **steel blue accent**, IBM Plex font stacks.
- Components: `.panel`, `.page-header`, `.result-card`, `.viz-sound-*`, `.embedding-viz*`, etc.
- **Viz** top line uses dark chrome; labels use mono stack variables.

---

## 14. Known limitations (honest list for planning)

| Area | Limitation |
|------|------------|
| Search | Substring only on static array; no pagination/facets. |
| Classify / similarity | No model; fixed outputs independent of audio. |
| Saved | Full JSON of rows in `localStorage` — size and privacy considerations. |
| Compare | In-memory only until refresh; not deep-linked. |
| Viz | Simplified physics of frequency estimation; bloom + additive art direction, not calibrated science viz. |
| A11y | Viz is WebGL-heavy; labels have some ARIA on container but deep accessibility not audited. |

---

## 15. File-to-feature quick index

| If you care about… | Open… |
|---------------------|--------|
| Routes / providers | `App.tsx` |
| Nav + footer | `layout/AppShell.tsx` |
| Mock HTTP-shaped API | `api/mock.ts` |
| Shared DTOs | `api/types.ts` |
| Upload + vocab persistence | `context/AppPreferences.tsx` |
| Starred list | `context/SavedContext.tsx`, `saved/savedStore.ts` |
| Spectrogram drawing | `lib/spectrogramCanvas.ts` |
| Frame features + chirp chains | `lib/audioDrivenPointCloud.ts` |
| 3D + bloom + narrative | `components/BirdSoundEmbeddingViz.tsx` |
| Global look | `index.css`, `index.html` (fonts) |

---

When you add a feature, append a subsection here (or add a linked doc) so the next developer does not reverse-engineer the UI.
