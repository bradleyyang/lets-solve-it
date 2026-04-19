import {
  createContext,
  useCallback,
  useContext,
  useMemo,
  useState,
  type ReactNode,
} from "react";
import type { SearchResult } from "@/api/types";
import { loadSaved, toggleSaved } from "@/saved/savedStore";

interface SavedContextValue {
  saved: SearchResult[];
  toggle: (r: SearchResult) => void;
  isSaved: (id: string) => boolean;
}

const SavedContext = createContext<SavedContextValue | null>(null);

export function SavedProvider({ children }: { children: ReactNode }) {
  const [version, setVersion] = useState(0);
  const saved = useMemo(() => loadSaved(), [version]);

  const toggle = useCallback((r: SearchResult) => {
    toggleSaved(r);
    setVersion((v) => v + 1);
  }, []);

  const isSaved = useCallback(
    (id: string) => {
      return saved.some((x) => x.id === id);
    },
    [saved]
  );

  const value = useMemo(
    () => ({ saved, toggle, isSaved }),
    [saved, toggle, isSaved]
  );

  return <SavedContext.Provider value={value}>{children}</SavedContext.Provider>;
}

export function useSaved(): SavedContextValue {
  const ctx = useContext(SavedContext);
  if (!ctx) throw new Error("useSaved must be used within SavedProvider");
  return ctx;
}
