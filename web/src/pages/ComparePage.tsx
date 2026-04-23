import { Link } from "react-router-dom";
import { getResultById } from "@/api/mock";
import { useAppPreferences } from "@/context/AppPreferences";
import { ResultCard } from "@/components/ResultCard";

export function ComparePage() {
  const { compareSlots, clearCompare } = useAppPreferences();
  const [a, b] = compareSlots;
  const ra = a ? getResultById(a) : undefined;
  const rb = b ? getResultById(b) : undefined;

  return (
    <div className="page compare-page">
      <header className="page-header">
        <h1>Paired comparison</h1>
        <p className="muted">
          Select two query results for side-by-side inspection. Slots are filled from the{" "}
          <strong>Compare</strong> action on each result card.
        </p>
      </header>
      {!ra && !rb ? (
        <section className="panel">
          <p className="muted panel__flush">
            No recordings in the comparison queue.{" "}
            <Link to="/query">Return to Query</Link> and assign two clips.
          </p>
        </section>
      ) : (
        <>
          <div className="compare-toolbar">
            <button type="button" className="btn btn--outline" onClick={clearCompare}>
              Clear slots
            </button>
          </div>
          <div className="compare-grid">
            <div>
              {!ra ? (
                <p className="compare-slot muted">Slot A · unassigned</p>
              ) : (
                <ResultCard result={ra} />
              )}
            </div>
            <div>
              {!rb ? (
                <p className="compare-slot muted">Slot B · select a second recording from Query</p>
              ) : (
                <ResultCard result={rb} />
              )}
            </div>
          </div>
        </>
      )}
    </div>
  );
}
