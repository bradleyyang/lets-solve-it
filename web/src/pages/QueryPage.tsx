import { useEffect, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { classifyUpload, searchDataset, searchSimilarToUpload } from "@/api/mock";
import type { ClassificationHit, SearchResult } from "@/api/types";
import { useAppPreferences } from "@/context/AppPreferences";
import { ResultCard } from "@/components/ResultCard";
import { useSpectrogram } from "@/hooks/useSpectrogram";

export function QueryPage() {
  const [params] = useSearchParams();
  const sourceHint = params.get("source");

  const {
    vocabMode,
    setVocabMode,
    uploadedFile,
    setUploadedFile,
  } = useAppPreferences();

  const [query, setQuery] = useState("");
  const [results, setResults] = useState<SearchResult[]>([]);
  const [loading, setLoading] = useState(false);
  const [classifyHits, setClassifyHits] = useState<ClassificationHit[] | null>(null);
  const [classifyLoading, setClassifyLoading] = useState(false);
  const [specCanvas, setSpecCanvas] = useState<HTMLCanvasElement | null>(null);

  useSpectrogram(uploadedFile, specCanvas);

  useEffect(() => {
    if (sourceHint !== "dataset") return;
    setQuery((q) => q || (vocabMode === "scientific" ? "Turdus" : "sparrow"));
  }, [sourceHint, vocabMode]);

  const runDatasetSearch = async () => {
    setLoading(true);
    try {
      const r = await searchDataset(query, vocabMode);
      setResults(r);
    } finally {
      setLoading(false);
    }
  };

  const runSimilarSearch = async () => {
    if (!uploadedFile) return;
    setLoading(true);
    try {
      const r = await searchSimilarToUpload(uploadedFile);
      setResults(r);
    } finally {
      setLoading(false);
    }
  };

  const runClassify = async () => {
    if (!uploadedFile) return;
    setClassifyLoading(true);
    try {
      const h = await classifyUpload(uploadedFile);
      setClassifyHits(h);
    } finally {
      setClassifyLoading(false);
    }
  };

  return (
    <div className="page query-page">
      <header className="page-header">
        <h1>Query workspace</h1>
        <p className="muted">
          Text retrieval against a static mock table, optional upload-driven classification scores,
          and mock acoustic similarity — all evaluated client-side.
        </p>
      </header>

      <section className="panel">
        <h2>Vocabulary</h2>
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
        <p className="muted small">
          Current mode affects how mock search matches strings (e.g. “Turdus” vs “robin”).
        </p>
      </section>

      <section className="panel">
        <h2>Catalog search</h2>
        <div className="row gap wrap">
          <input
            type="search"
            className="input"
            placeholder={
              vocabMode === "scientific"
                ? "Try Turdus, Poecile, …"
                : "Try sparrow, chickadee, …"
            }
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && runDatasetSearch()}
          />
          <button type="button" className="btn btn--primary" onClick={runDatasetSearch} disabled={loading}>
            {loading ? "Searching…" : "Search dataset"}
          </button>
        </div>
      </section>

      <section className="panel">
        <h2>Upload &amp; classifier output</h2>
        <div className="row gap wrap">
          <label className="file-input">
            <input
              type="file"
              accept="audio/*,.mp3,.wav,.ogg,.webm,.m4a"
              onChange={(e) => {
                const f = e.target.files?.[0];
                setUploadedFile(f ?? null);
                setClassifyHits(null);
              }}
            />
            <span className="btn btn--outline">Choose audio</span>
          </label>
          {uploadedFile ? <span className="file-name">{uploadedFile.name}</span> : null}
          <button
            type="button"
            className="btn btn--outline"
            disabled={!uploadedFile || classifyLoading}
            onClick={runClassify}
          >
            {classifyLoading ? "Classifying…" : "Run mock classification"}
          </button>
          <button
            type="button"
            className="btn btn--primary"
            disabled={!uploadedFile || loading}
            onClick={runSimilarSearch}
          >
            {loading ? "Searching…" : "Search similar (mock)"}
          </button>
        </div>
        <div className="spectrogram-preview">
          <canvas ref={setSpecCanvas} width={320} height={96} aria-label="Upload spectrogram" />
        </div>
        {classifyHits ? (
          <div className="classify-hits">
            <h3>Posterior over labels (mock)</h3>
            <ol>
              {classifyHits.map((h) => (
                <li key={h.label}>
                  <strong>{h.label}</strong>
                  {h.scientificName ? (
                    <span className="muted"> — {h.scientificName}</span>
                  ) : null}{" "}
                  <span className="score">{(h.score * 100).toFixed(1)}%</span>
                </li>
              ))}
            </ol>
          </div>
        ) : null}
      </section>

      <section className="panel">
        <div className="row spread">
          <h2>Result set</h2>
          <Link to="/saved" className="muted small">
            Saved list →
          </Link>
        </div>
        {results.length === 0 ? (
          <p className="muted">Run a dataset search or similarity search to see cards here.</p>
        ) : (
          <div className="results-grid">
            {results.map((r) => (
              <ResultCard key={r.id} result={r} />
            ))}
          </div>
        )}
      </section>
    </div>
  );
}
