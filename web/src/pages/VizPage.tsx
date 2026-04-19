import { Link, useParams } from "react-router-dom";
import { getResultById } from "@/api/mock";

export function VizPage() {
  const { id } = useParams<{ id: string }>();
  const result = id ? getResultById(id) : undefined;

  return (
    <div className="page viz-page">
      <p className="breadcrumb">
        <Link to="/query">← Back to query</Link>
      </p>
      <h1>3D sound visualization</h1>
      {result ? (
        <p className="muted">
          Placeholder view for recording <strong>{result.recordingId}</strong> —{" "}
          {result.commonName}. Replace this route with WebGL / Three.js when you wire real audio.
        </p>
      ) : (
        <p className="muted">Unknown recording id.</p>
      )}
      <div className="viz-placeholder" aria-hidden>
        <div className="viz-placeholder__orb" />
        <p className="viz-placeholder__caption">Mock 3D ambience</p>
      </div>
    </div>
  );
}
