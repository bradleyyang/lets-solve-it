import {
  createContext,
  useCallback,
  useContext,
  useMemo,
  useState,
  type ReactNode,
} from "react";
import type { VocabMode } from "@/api/types";

const VOCAB_KEY = "lets-solve-it:vocab";

function readVocab(): VocabMode {
  try {
    const v = localStorage.getItem(VOCAB_KEY);
    if (v === "scientific" || v === "common") return v;
  } catch {
    /* ignore */
  }
  return "common";
}

function writeVocab(v: VocabMode): void {
  try {
    localStorage.setItem(VOCAB_KEY, v);
  } catch {
    /* ignore */
  }
}

type CompareSlot = string | null;

export type CompareAddResult =
  | "duplicate"
  | "added_first"
  | "added_second"
  | "replaced";

interface AppPreferencesValue {
  vocabMode: VocabMode;
  setVocabMode: (v: VocabMode) => void;
  uploadedFile: File | null;
  setUploadedFile: (f: File | null) => void;
  compareSlots: [CompareSlot, CompareSlot];
  addToCompare: (id: string) => CompareAddResult;
  clearCompare: () => void;
}

const AppPreferencesContext = createContext<AppPreferencesValue | null>(null);

export function AppPreferencesProvider({ children }: { children: ReactNode }) {
  const [vocabMode, setVocabModeState] = useState<VocabMode>(() => readVocab());
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [compareSlots, setCompareSlots] = useState<[CompareSlot, CompareSlot]>([
    null,
    null,
  ]);

  const setVocabMode = useCallback((v: VocabMode) => {
    setVocabModeState(v);
    writeVocab(v);
  }, []);

  const addToCompare = useCallback((id: string): CompareAddResult => {
    let out: CompareAddResult = "added_first";
    setCompareSlots(([a, b]) => {
      if (a === id || b === id) {
        out = "duplicate";
        return [a, b];
      }
      if (!a) {
        out = "added_first";
        return [id, b];
      }
      if (!b) {
        out = "added_second";
        return [a, id];
      }
      out = "replaced";
      return [id, b];
    });
    return out;
  }, []);

  const clearCompare = useCallback(() => setCompareSlots([null, null]), []);

  const value = useMemo(
    () => ({
      vocabMode,
      setVocabMode,
      uploadedFile,
      setUploadedFile,
      compareSlots,
      addToCompare,
      clearCompare,
    }),
    [
      vocabMode,
      setVocabMode,
      uploadedFile,
      compareSlots,
      addToCompare,
      clearCompare,
    ]
  );

  return (
    <AppPreferencesContext.Provider value={value}>
      {children}
    </AppPreferencesContext.Provider>
  );
}

export function useAppPreferences(): AppPreferencesValue {
  const ctx = useContext(AppPreferencesContext);
  if (!ctx) {
    throw new Error("useAppPreferences must be used within AppPreferencesProvider");
  }
  return ctx;
}
