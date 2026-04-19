import { NavLink, Outlet } from "react-router-dom";
import { useAppPreferences } from "@/context/AppPreferences";

export function AppShell() {
  const { compareSlots } = useAppPreferences();
  const compareReady = compareSlots[0] && compareSlots[1];

  return (
    <div className="app-shell">
      <header className="app-shell__header">
        <div className="app-shell__brand">
          <span className="app-shell__logo" aria-hidden>
            ◈
          </span>
          <span className="app-shell__title">Bird audio lab</span>
        </div>
        <nav className="app-shell__nav" aria-label="Main">
          <NavLink
            to="/"
            end
            className={({ isActive }) =>
              isActive ? "nav-link nav-link--active" : "nav-link"
            }
          >
            Home
          </NavLink>
          <NavLink
            to="/query"
            className={({ isActive }) =>
              isActive ? "nav-link nav-link--active" : "nav-link"
            }
          >
            Query
          </NavLink>
          <NavLink
            to="/saved"
            className={({ isActive }) =>
              isActive ? "nav-link nav-link--active" : "nav-link"
            }
          >
            Saved
          </NavLink>
          <NavLink
            to="/compare"
            className={({ isActive }) =>
              isActive ? "nav-link nav-link--active" : "nav-link"
            }
          >
            Compare
            {compareReady ? (
              <span className="nav-badge" title="Two clips selected">
                2
              </span>
            ) : null}
          </NavLink>
        </nav>
      </header>
      <main className="app-shell__main">
        <Outlet />
      </main>
    </div>
  );
}
