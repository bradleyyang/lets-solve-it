import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom";
import { AppPreferencesProvider } from "@/context/AppPreferences";
import { SavedProvider } from "@/context/SavedContext";
import { AppShell } from "@/layout/AppShell";
import { HomePage } from "@/pages/HomePage";
import { QueryPage } from "@/pages/QueryPage";
import { SavedPage } from "@/pages/SavedPage";
import { VizPage } from "@/pages/VizPage";
import { ComparePage } from "@/pages/ComparePage";

export default function App() {
  return (
    <BrowserRouter>
      <AppPreferencesProvider>
        <SavedProvider>
          <Routes>
            <Route element={<AppShell />}>
              <Route path="/" element={<HomePage />} />
              <Route path="/query" element={<QueryPage />} />
              <Route path="/saved" element={<SavedPage />} />
              <Route path="/compare" element={<ComparePage />} />
              <Route path="/viz/:id" element={<VizPage />} />
            </Route>
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </SavedProvider>
      </AppPreferencesProvider>
    </BrowserRouter>
  );
}
