import { Suspense, lazy } from "react";
import { Link, useParams } from "react-router-dom";
import { getResultById } from "@/api/mock";
import { useAppPreferences } from "@/context/AppPreferences";

const BirdSoundEmbeddingViz = lazy(async () => {
  const m = await import("@/components/BirdSoundEmbeddingViz");
  return { default: m.BirdSoundEmbeddingViz };
});

export function VizPage() {
  const { id } = useParams<{ id: string }>();
  const result = id ? getResultById(id) : undefined;
  const { uploadedFile } = useAppPreferences();

  return (
    <div className="page viz-page">
      <p className="breadcrumb">
        <Link to="/query">← Back to query</Link>
      </p>
      <header className="page-header">
        <h1>3D sound stream</h1>
        <p className="muted">
          Frame-level audio analysis sampled at 60 fps. Each emitted point carries amplitude + dominant frequency and
          evolves in a generative 3D network over its lifetime.
        </p>
      </header>

      {result ? (
        <>
          <div className="viz-sound-stack">
            <div className="viz-sound-panel viz-sound-panel--stage">
              <div className="embedding-viz-topline">
                <span className="embedding-viz-stats">Preparing stream…</span>
                <span className="embedding-viz-source">
                  Source: {uploadedFile ? `uploaded file (${uploadedFile.name})` : "synthetic fallback"}
                </span>
              </div>
              <Suspense
                fallback={
                  <div className="embedding-viz embedding-viz--loading muted" aria-busy="true">
                    Loading WebGL view…
                  </div>
                }
              >
                <BirdSoundEmbeddingViz seed={`${result.id}:${result.recordingId}`} audioFile={uploadedFile} />
              </Suspense>
            </div>
            <footer className="viz-sound-species-card">
              <img
                className="viz-sound-species-card__thumb"
                src={result.imageUrl}
                alt=""
                width={80}
                height={80}
                loading="lazy"
              />
              <div className="viz-sound-species-card__meta">
                <div className="viz-sound-species-card__title">{result.commonName}</div>
                <div className="viz-sound-species-card__sci muted">{result.scientificName}</div>
                <div className="viz-sound-species-card__rec small muted">
                  Recording {result.recordingId} · {result.vocalizationType}
                </div>
              </div>
            </footer>
          </div>
          <section className="viz-data-guide">
            <h2>Data for each point</h2>
            <div className="viz-data-guide__bar" aria-hidden>
              <span>2 kHz</span>
              <span>3 kHz</span>
              <span>4 kHz</span>
              <span>5 kHz</span>
              <span>6 kHz</span>
              <span>7 kHz</span>
              <span>8 kHz</span>
            </div>
            <p className="small muted">
              Color = dominant frequency band at each frame · Numbers = amplitude and time for the lit chain only ·
              The view starts empty; each detected chirp lights a chain along time, then it fades away.
            </p>
          </section>
          <p className="muted small viz-page__hint">
            Drag to orbit · scroll to zoom · right-drag to pan. Nothing is shown until a chirp; then squares light in
            sequence along the chain and slowly disappear before the next chirp.
          </p>
        </>
      ) : (
        <p className="muted">Unknown recording id.</p>
      )}
    </div>
  );
}
