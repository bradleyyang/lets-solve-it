import { ResultCard } from "@/components/ResultCard";
import { useSaved } from "@/context/SavedContext";

export function SavedPage() {
  const { saved } = useSaved();

  return (
    <div className="page saved-page">
      <header className="page-header">
        <h1>Saved specimens</h1>
        <p className="muted">
          Subset of catalog rows marked for follow-up on the Query view. Data live only in this
          browser profile.
        </p>
      </header>
      {saved.length === 0 ? (
        <p className="muted">Nothing saved yet. Run a search and press Save on a card.</p>
      ) : (
        <div className="results-grid">
          {saved.map((r) => (
            <ResultCard key={r.id} result={r} />
          ))}
        </div>
      )}
    </div>
  );
}
