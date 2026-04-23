# Web application — documentation

This folder is the **onboarding entry point** for anyone working on the `web/` Vite + React client. The Python/ML pipeline and Xeno-canto tooling live at the **repository root**; see [`CODEBASE_GUIDE.md`](../../CODEBASE_GUIDE.md) there.

## Read order for a new web developer

1. **[FEATURES.md](./FEATURES.md)** — What the app does today: every screen, user flow, mock API, and major component (including the 3D viz).
2. **[DEVELOPMENT.md](./DEVELOPMENT.md)** — How to run, build, path aliases, and where to change things safely.

## Scope (important)

- **No backend** is required: search, classification, and similarity are **mocked in the browser**.
- **Persistence** is only `localStorage` (saved list + vocabulary preference). Uploads live in React state until refresh.
- The **temporal embedding view** uses real Web Audio decoding when you pass a file; otherwise it uses a deterministic synthetic signal for demos.

## Source layout (quick map)

| Path | Role |
|------|------|
| `src/App.tsx` | Router + providers |
| `src/layout/AppShell.tsx` | Header, nav, footer |
| `src/pages/*` | Route-level screens |
| `src/components/` | `ResultCard`, lazy `BirdSoundEmbeddingViz` |
| `src/api/` | Types + `mock.ts` (replace with real API later) |
| `src/context/` | Preferences + saved list |
| `src/saved/` | `localStorage` read/write for saved rows |
| `src/lib/` | Spectrogram + audio-driven point cloud math |
| `src/hooks/` | `useSpectrogram` |
| `src/index.css` | Global + page styles |

Questions about **training data, CLAP scripts, or CSV schema** belong in the root guide and `docs/` under the repo root, not here.
