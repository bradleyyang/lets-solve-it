import type { SearchResult } from "@/api/types";

const STORAGE_KEY = "lets-solve-it:saved";

function readRaw(): string | null {
  try {
    return localStorage.getItem(STORAGE_KEY);
  } catch {
    return null;
  }
}

export function loadSaved(): SearchResult[] {
  const raw = readRaw();
  if (!raw) return [];
  try {
    const parsed = JSON.parse(raw) as unknown;
    if (!Array.isArray(parsed)) return [];
    return parsed as SearchResult[];
  } catch {
    return [];
  }
}

function write(items: SearchResult[]): void {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(items));
  } catch {
    /* ignore quota */
  }
}

export function saveAll(items: SearchResult[]): void {
  write(items);
}

export function toggleSaved(result: SearchResult): SearchResult[] {
  const cur = loadSaved();
  const idx = cur.findIndex((r) => r.id === result.id);
  if (idx >= 0) {
    cur.splice(idx, 1);
  } else {
    cur.unshift(result);
  }
  write(cur);
  return cur;
}
