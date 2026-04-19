import { useEffect, useState } from "react";
import { drawSpectrogramFromFile } from "@/lib/spectrogramCanvas";

export function useSpectrogram(file: File | null, canvas: HTMLCanvasElement | null) {
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!file || !canvas) {
      setError(null);
      setLoading(false);
      return;
    }
    let cancelled = false;
    setLoading(true);
    setError(null);
    drawSpectrogramFromFile(canvas, file)
      .then(() => {
        if (!cancelled) setLoading(false);
      })
      .catch(() => {
        if (!cancelled) {
          setError("Could not decode audio for spectrogram.");
          setLoading(false);
        }
      });
    return () => {
      cancelled = true;
    };
  }, [file, canvas]);

  return { loading, error };
}
