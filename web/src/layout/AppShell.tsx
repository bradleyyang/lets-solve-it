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
            <svg width="28" height="28" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
              <path
                d="M4 14c4.5-5 10.5-6.5 17-4.5-2 2.5-2.5 6-1 9-4.5-2.5-10-3-14 2a18 18 0 0 1-2-6.5Z"
                fill="currentColor"
                opacity="0.92"
              />
              <path
                d="M14.5 11c3-4 7.5-5.5 11-4"
                stroke="currentColor"
                strokeWidth="1.75"
                strokeLinecap="round"
                opacity="0.85"
              />
              <circle cx="7.5" cy="12.5" r="1.1" fill="#fff" opacity="0.95" />
            </svg>
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
