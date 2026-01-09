"""Run both frontend (static) and backend (FastAPI) from a single file.

Usage:
    python run_all.py

This will mount the first existing directory among
`frontend`, `app`, `public`, `static`, `dist`, `build` at `/`.
If none are found, a small demo HTML page is served at `/`.
The backend API from `predict_api.py` is used for `/api/*` routes.
"""
import os
from pathlib import Path
import uvicorn
import webbrowser
import threading
import time
import urllib.request
import subprocess
import http.server
import socketserver

try:
    # import the existing FastAPI app from predict_api
    from predict_api import app
except Exception as e:
    raise RuntimeError("Failed to import `predict_api.app`. Ensure predict_api.py is present and importable.")

from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse


FRONTEND_CANDIDATES = ["frontend", "app", "public", "static", "dist", "build"]


def find_frontend_dir():
    for d in FRONTEND_CANDIDATES:
        p = Path(d)
        if p.exists() and p.is_dir():
            return str(p.resolve())
    return None


def mount_frontend(app):
    frontend_dir = find_frontend_dir()
    if frontend_dir:
        app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
        print(f"Mounted static frontend from: {frontend_dir}")
    else:
        print("No frontend directory found — serving minimal demo page at /")

        @app.get("/", response_class=HTMLResponse)
        async def index():
            return HTMLResponse(
                """
                <!doctype html>
                <html>
                <head><meta charset="utf-8"><title>Heart Disease Predictor (Demo)</title></head>
                <body>
                  <h2>Heart Disease Predictor (Demo)</h2>
                  <p>This is a minimal demo page. It calls <code>/api/predict</code> with example data.</p>
                  <button id="run">Run demo prediction</button>
                  <pre id="out"></pre>
                  <script>
                    document.getElementById('run').onclick = async () => {
                      const payload = {
                        age: 55, sex: 1, bmi: 27, systolic_bp: 130, diastolic_bp: 82,
                        heart_rate: 75, prevalent_hypertension: 0, total_cholesterol: 200,
                        hdl:50, ldl:120, triglycerides:150, fasting_glucose:95, diabetes:0,
                        sodium:140, potassium:4.2, calcium:9.5, creatinine:1.0, egfr:90,
                        smoking:0, physical_activity:1, family_history:0
                      };
                      try {
                        const res = await fetch('/api/predict', {
                          method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify(payload)
                        });
                        const data = await res.json();
                        document.getElementById('out').textContent = JSON.stringify(data, null, 2);
                      } catch(err) {
                        document.getElementById('out').textContent = 'Error: '+err;
                      }
                    };
                  </script>
                </body>
                </html>
                """
            )


def main(host: str = "0.0.0.0", port: int = 8000, frontend_port: int = 3000, open_browser: bool = True):
    # Do not mount the raw `frontend` dir eagerly; decide below whether to
    # mount a built `dist/` into FastAPI root or start a dev/static server.
    # mount_frontend(app)

    bind_url = f"http://{host}:{port}"
    # browsers cannot open 0.0.0.0; prefer localhost for the client URL
    backend_open_url = f"http://127.0.0.1:{port}" if host == "0.0.0.0" else bind_url
    frontend_dir = find_frontend_dir()
    frontend_server = None
    frontend_open_url = None
    if frontend_dir:
        # If a built bundle exists, serve it. Otherwise, if a package.json
        # exists assume a dev workflow (Vite) and start `npm run dev`.
        dist_dir = os.path.join(frontend_dir, 'dist')
        build_dir = os.path.join(frontend_dir, 'build')
        pkg_json = os.path.join(frontend_dir, 'package.json')

        if os.path.exists(dist_dir) or os.path.exists(build_dir):
            # Prefer mounting the built frontend into the FastAPI app so API
            # requests (`/api/*`) are handled by the backend rather than
            # hitting a separate static server which doesn't accept POST.
            serve_dir = dist_dir if os.path.exists(dist_dir) else build_dir
            app.mount('/', StaticFiles(directory=serve_dir, html=True), name='frontend')
            frontend_open_url = backend_open_url
            print(f"Mounted built frontend at FastAPI root; serving on {frontend_open_url}")
        elif os.path.exists(pkg_json):
            # Start frontend dev server via npm (assumes `npm install` was run)
            # Use configured `frontend_port` so the polling/open logic matches the dev server.
            frontend_open_url = f'http://127.0.0.1:{frontend_port}'
            print(f"Found package.json; will attempt to start dev server and open {frontend_open_url}")
        else:
            # Fallback: mount the frontend directory into FastAPI so API POSTs
            # to `/api/*` are handled by the backend (avoid static-only server)
            app.mount('/', StaticFiles(directory=frontend_dir, html=True), name='frontend')
            frontend_open_url = backend_open_url
            print(f"Mounted frontend dir at FastAPI root; serving on {frontend_open_url}")
    else:
        frontend_open_url = backend_open_url
    print(f"Server binding: {bind_url}")
    print(f"Open URL: {frontend_open_url}")

    # If a frontend directory exists, either start the dev server (npm) or
    # serve the built static files.
    frontend_proc = None
    frontend_thread = None
    if frontend_dir:
        dist_dir = os.path.join(frontend_dir, 'dist')
        build_dir = os.path.join(frontend_dir, 'build')
        pkg_json = os.path.join(frontend_dir, 'package.json')

        if os.path.exists(dist_dir) or os.path.exists(build_dir):
            # Already mounted into FastAPI above; nothing to start separately.
            pass

        elif os.path.exists(pkg_json):
            # Launch `npm run dev` in the frontend directory. Pass PORT in env
            # so Vite will bind to `frontend_port` configured above.
            try:
                print("Starting frontend dev server (npm run dev)...")
                env = os.environ.copy()
                env['PORT'] = str(frontend_port)
                frontend_proc = subprocess.Popen(['npm', 'run', 'dev'], cwd=frontend_dir, env=env)
            except FileNotFoundError:
                print("npm not found. Please ensure Node.js and npm are installed.")

        else:
            # Already mounted into FastAPI above in the else-branch; nothing to start.
            pass

    # Start uvicorn in a background thread (avoids subprocess import issues).
    def run_server():
        uvicorn.run(app, host=host, port=port)

    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()

    # Poll for readiness
    timeout = 15.0
    interval = 0.25
    elapsed = 0.0
    ready = False
    while elapsed < timeout:
        try:
            with urllib.request.urlopen(frontend_open_url, timeout=1) as resp:
                if resp.status in (200, 204, 301, 302):
                    ready = True
                    break
        except Exception:
            pass
        time.sleep(interval)
        elapsed += interval

    if ready and open_browser:
        try:
            webbrowser.open(frontend_open_url)
        except Exception:
            pass

    try:
        # Poll frontend URL until available (or timeout)
        timeout = 30.0
        interval = 0.25
        elapsed = 0.0
        ready = False
        while elapsed < timeout:
            try:
                with urllib.request.urlopen(frontend_open_url, timeout=1) as resp:
                    if resp.status in (200, 204, 301, 302):
                        ready = True
                        break
            except Exception:
                pass
            time.sleep(interval)
            elapsed += interval

        if ready and open_browser:
            try:
                webbrowser.open(frontend_open_url)
            except Exception:
                pass

        # Keep main thread alive while server runs
        while server_thread.is_alive():
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("Shutting down")
    finally:
        if frontend_proc:
            try:
                frontend_proc.terminate()
            except Exception:
                pass

if __name__ == '__main__':
    # Change to the script's directory so relative paths work correctly
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    print(f"Working directory: {os.getcwd()}")
    
    # Allow disabling auto browser open via env var
    open_browser = os.environ.get('RUN_ALL_OPEN_BROWSER', '1') != '0'
    main(open_browser=open_browser)
