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
      <header className="page-header">
        <h1>Dashboard</h1>
        <p className="muted">
          Explore uploads, search the (mock) indexed dataset, and switch between common names and
          scientific names — all wired for a future CLAP backend.
        </p>
      </header>

      <section className="panel">
        <h2>Upload audio</h2>
        <p className="muted">
          Select a recording; we decode it in the browser and draw a spectrogram preview — no
          server required.
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
            Continue to Query with this clip
          </Link>
        </p>
      </section>

      <section className="panel">
        <h2>Search the dataset</h2>
        <p className="muted">
          Run text queries against a mocked catalog of Xeno-canto–style rows (species, vocalization,
          duration).
        </p>
        <Link to="/query?source=dataset" className="btn btn--primary">
          Open dataset search
        </Link>
      </section>

      <section className="panel">
        <h2>Classification + query</h2>
        <p className="muted">
          On the Query page, upload a clip to see mock top‑k species scores, then search for similar
          mock hits.
        </p>
        <Link to="/query" className="btn btn--outline">
          Go to Query
        </Link>
      </section>

      <section className="panel">
        <h2>Vocabulary mode</h2>
        <p className="muted">
          Prefer natural language names or scientific names when searching — stored in your browser.
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
