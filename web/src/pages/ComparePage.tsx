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
      <h1>Compare clips</h1>
      <p className="muted">
        Pick two results from the query grid using <strong>Compare</strong>, or choose another
        recording to replace the first slot.
      </p>
      {!ra && !rb ? (
        <p>
          Nothing selected yet.{" "}
          <Link to="/query">Go to Query</Link> and add clips to compare.
        </p>
      ) : (
        <>
          <div className="compare-toolbar">
            <button type="button" className="btn btn--outline" onClick={clearCompare}>
              Clear selection
            </button>
          </div>
          <div className="compare-grid">
            <div>
              {!ra ? (
                <p className="compare-slot muted">Slot 1 — empty</p>
              ) : (
                <ResultCard result={ra} />
              )}
            </div>
            <div>
              {!rb ? (
                <p className="compare-slot muted">Slot 2 — add a second clip from Query</p>
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
