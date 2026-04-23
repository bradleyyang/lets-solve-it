# Web — development guide

## Prerequisites

- **Node.js** 18+ (LTS recommended)
- **npm** (comes with Node)

## Commands

From the `web/` directory:

```bash
npm install          # once per clone / after dependency changes
npm run dev          # Vite dev server (HMR)
npm run build        # TypeScript check + production bundle
npm run preview      # Serve the production build locally
```

There is no separate `lint` script in `package.json` today; `npm run build` runs `tsc --noEmit` and will catch type errors.

## Path alias

- **`@/`** → `web/src/` (configured in `vite.config.ts`)

Imports look like: `import { x } from "@/api/mock"`.

## Adding a new route

1. Add a `<Route>` under the existing `<Route element={<AppShell />}>` in `src/App.tsx`.
2. Create a page component under `src/pages/`.
3. Add a `<NavLink>` in `src/layout/AppShell.tsx` if it should appear in the main nav.

## Environment variables

The web client **does not** read `.env` for Xeno-canto or ML today. Any future `VITE_*` variables must be prefixed for Vite exposure and documented here when introduced.

## Dependencies worth knowing

| Package | Use |
|---------|-----|
| `react`, `react-dom` | UI |
| `react-router-dom` | SPA routing |
| `three` | WebGL + postprocessing (viz chunk) |

## Production build notes

- **`BirdSoundEmbeddingViz`** is **lazy-loaded** from `VizPage.tsx` to keep the initial bundle smaller; editing it affects a separate chunk.
- Build may warn about chunk size for the viz bundle; that is expected while Three + postprocessing are bundled together.

## Styling

- Global rules and page-specific classes live in **`src/index.css`** (no CSS-in-JS, no Tailwind).
- Prefer reusing existing utility patterns: `.panel`, `.page-header`, `.muted`, `.btn`, etc.

## Replacing mocks with a real API

All network-shaped behaviour today goes through **`src/api/mock.ts`**. A practical migration:

1. Introduce `src/api/client.ts` (or similar) with `fetch` to your backend.
2. Keep **`src/api/types.ts`** as the contract between UI and server where possible.
3. Swap `QueryPage` / `VizPage` / `ComparePage` imports from `mock` to the new module, or add a thin factory keyed by env.

Do not remove mock types until the server returns a compatible shape (or add adapters).
