export type VocabMode = "common" | "scientific";

/** Where the search “signal” comes from (mock only). */
export type QuerySource = "dataset" | "upload";

export interface ClassificationHit {
  label: string;
  scientificName?: string;
  score: number;
}

export interface SearchResult {
  id: string;
  recordingId: string;
  commonName: string;
  scientificName: string;
  vocalizationType: string;
  duration: string;
  speciesCode: string;
  similarity?: number;
  /** Placeholder image (deterministic mock). */
  imageUrl: string;
}
