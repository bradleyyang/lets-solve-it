# Bird audio dashboard (web)

## Prerequisites

You need **Node.js** (LTS recommended), which includes **npm**. Install from [https://nodejs.org/](https://nodejs.org/) and use the option to add Node to your PATH when the installer offers it.

After installing, **close and reopen** your terminal (and Cursor’s integrated terminal), then verify:

```powershell
node -v
npm -v
```

Both commands should print version numbers.

## Install and run

```powershell
cd web
npm install
npm run dev
```

Then open the URL Vite prints (usually `http://localhost:5173`).

## Troubleshooting

### `npm` / `node` is not recognized (PowerShell or CMD)

- Node is missing or not on `PATH`. Install Node.js LTS from the link above, restart the terminal, and try again.
- If Node is already installed, ensure your user **PATH** includes the folder that contains `node.exe` and `npm.cmd` (often `C:\Program Files\nodejs\`). Restart the terminal after changing environment variables.
- If you use **nvm-windows**, open a new shell and run `nvm use <version>` so `node` and `npm` are available in that session.

### Other install errors

- **Permission errors:** run the terminal as a normal user (not required to run as Administrator for `npm install` in a project folder).
- **Corporate proxy / SSL:** configure npm proxy or registry settings per your network policy.
- **Clear cache:** `npm cache clean --force` then `npm install` again (use sparingly).
