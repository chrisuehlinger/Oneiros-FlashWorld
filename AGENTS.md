# VR Dream Prototype – Code Map

This repository now consists of three cooperating pieces:

1. **`backend/flashworld_runner.py`** – lazily loads FlashWorld, exports dataclasses (`CameraPose`, `GenerationResult`) and provides the `FlashWorldRunner` API (initial camera rig, prompt image prep, `.spz` and frame extraction, seed handoff).
2. **`backend/server.py`** – aiohttp WebSocket service that bootstraps a session, streams frame-zero seed images/poses, manages pose windows, and exposes static assets:
   - `/ws`  WebSocket (poseBatch ↔ generation/bootstrap)
   - `/assets/` Generated outputs (`gaussians.spz`, frames, seed)
   - `/frontend/` Static frontend module files
   - `/`  `frontend/index.html`
3. **`frontend/`**
   - `index.html` Full screen HUD, pointer-lock overlay, import map (Three.js 0.178 + Spark 0.1.10)
   - `vrdream.js` Three.js scene + SparkRenderer, pose sampling, WebSocket plumbing, crossfades between `SplatMesh` instances.

### Typical dev loop

```bash
python -m backend.server --device cuda:0 --offload-t5 --port 8876
# browse to http://127.0.0.1:8876/
```

HUD shows connection state, pending poses, and most recent generation latency; “Enter Dreamscape” grabs pointer lock for WASD navigation. Pose batches are pushed automatically once the backend finishes each generation. Crossfade logic swaps the rendered mesh without dropping frames.
