import { ResultCard } from "@/components/ResultCard";
import { useSaved } from "@/context/SavedContext";

export function SavedPage() {
  const { saved } = useSaved();

  return (
    <div className="page saved-page">
      <header className="page-header">
        <h1>Saved</h1>
        <p className="muted">Recordings you starred on the Query page (stored in this browser).</p>
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
