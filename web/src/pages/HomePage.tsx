import { Link } from "react-router-dom";
import { useState } from "react";
import { useAppPreferences } from "@/context/AppPreferences";
import { useSpectrogram } from "@/hooks/useSpectrogram";

export function HomePage() {
  const { vocabMode, setVocabMode, uploadedFile, setUploadedFile } = useAppPreferences();
  const [specCanvas, setSpecCanvas] = useState<HTMLCanvasElement | null>(null);
  useSpectrogram(uploadedFile, specCanvas);

  return (
    <div className="page home-page">
      <p className="instrument-strip">
        Prototype workspace · mock dataset · no server round-trip
      </p>
      <header className="page-header">
        <h1>Overview</h1>
        <p className="muted">
          Use this workspace to inspect uploads, run text search against a simulated catalog, and
          switch taxonomic labels. Behaviour is deterministic in-browser; a future service could
          replace the mock APIs without changing navigation.
        </p>
      </header>

      <section className="panel">
        <h2>Audio intake</h2>
        <p className="muted">
          Select a recording. The waveform is decoded locally and a coarse spectrogram is rendered
          for orientation only.
        </p>
        <div className="row gap">
          <label className="file-input">
            <input
              type="file"
              accept="audio/*,.mp3,.wav,.ogg,.webm,.m4a"
              onChange={(e) => {
                const f = e.target.files?.[0];
                setUploadedFile(f ?? null);
              }}
            />
            <span className="btn btn--outline">Choose file</span>
          </label>
          {uploadedFile ? (
            <span className="file-name">{uploadedFile.name}</span>
          ) : null}
        </div>
        <div className="spectrogram-preview">
          <canvas ref={setSpecCanvas} width={320} height={96} aria-label="Spectrogram preview" />
          {!uploadedFile ? (
            <p className="muted spectrogram-preview__hint">Spectrogram appears after you choose a file.</p>
          ) : null}
        </div>
        <p>
          <Link to="/query" className="btn btn--primary">
            Open query with this clip
          </Link>
        </p>
      </section>

      <section className="panel">
        <h2>Catalog search</h2>
        <p className="muted">
          Run substring queries against a static mock table (species label, vocalization class,
          duration metadata).
        </p>
        <Link to="/query?source=dataset" className="btn btn--primary">
          Dataset search
        </Link>
      </section>

      <section className="panel">
        <h2>Classification &amp; similarity</h2>
        <p className="muted">
          On the Query view, an uploaded clip can drive a fixed mock classifier and a mock
          nearest-neighbour list over the same catalog.
        </p>
        <Link to="/query" className="btn btn--outline">
          Query workspace
        </Link>
      </section>

      <section className="panel">
        <h2>Taxonomic display</h2>
        <p className="muted">
          Prefer common or scientific names in search strings. Preference is persisted in{" "}
          <code className="doc-code">localStorage</code>.
        </p>
        <div className="segmented" role="group" aria-label="Vocabulary mode">
          <button
            type="button"
            className={vocabMode === "common" ? "segmented__btn is-on" : "segmented__btn"}
            onClick={() => setVocabMode("common")}
          >
            Common names
          </button>
          <button
            type="button"
            className={vocabMode === "scientific" ? "segmented__btn is-on" : "segmented__btn"}
            onClick={() => setVocabMode("scientific")}
          >
            Scientific names
          </button>
        </div>
      </section>
    </div>
  );
}
