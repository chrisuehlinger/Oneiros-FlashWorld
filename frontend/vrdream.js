import * as THREE from "three";
import { PointerLockControls } from "three/examples/jsm/controls/PointerLockControls.js";
import { SparkRenderer, SplatMesh } from "@sparkjsdev/spark";

const MOVE_SPEED = 2.5;
const CROSSFADE_MS = 1000;

const hudEl = document.getElementById("hud");
const statusEl = document.getElementById("hud-status");
const pendingEl = document.getElementById("hud-pending");
const latencyEl = document.getElementById("hud-latency");
const windowEl = document.getElementById("hud-window");
const overlayEl = document.getElementById("instruction-overlay");
const startButton = document.getElementById("start-button");

const state = {
  renderer: null,
  camera: null,
  scene: null,
  controls: null,
  spark: null,
  splatGroup: null,
  currentMesh: null,
  fadePromise: Promise.resolve(),
  clock: new THREE.Clock(),
  socket: null,
  poseIntervalSeconds: 2,
  samplerInterval: null,
  pendingPoses: [],
  generationInFlight: false,
  lastGenerationId: null,
  lastLatencyMs: null,
  connectionStatus: "Initialising",
  windowSize: null,
  pointerLocked: false,
  errorMessage: null,
  movement: {
    forward: false,
    backward: false,
    left: false,
    right: false,
    up: false,
    down: false,
  },
};

function updateHUD() {
  if (!statusEl) return;

  const generationFlag = state.generationInFlight ? " (generation running)" : "";
  const errorSuffix = state.errorMessage ? ` — ${state.errorMessage}` : "";
  statusEl.textContent = `Status: ${state.connectionStatus}${generationFlag}${errorSuffix}`;

  pendingEl.textContent = `Pending poses: ${state.pendingPoses.length}`;
  latencyEl.textContent =
    state.lastLatencyMs != null
      ? `Generation latency: ${state.lastLatencyMs} ms`
      : "Generation latency: —";
  windowEl.textContent =
    state.windowSize != null ? `Window size: ${state.windowSize}` : "Window size: —";
}

function setHUDVisible(visible) {
  hudEl?.classList.toggle("hidden", !visible);
}

function hideOverlay() {
  overlayEl?.classList.add("hidden");
}

function showOverlay() {
  overlayEl?.classList.remove("hidden");
}

function setupScene() {
  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.outputEncoding = THREE.sRGBEncoding;
  document.body.appendChild(renderer.domElement);

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x05070f);

  const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
  camera.position.set(0, 1.6, 4);
  camera.lookAt(new THREE.Vector3(0, 1.3, 0));

  const controls = new PointerLockControls(camera, renderer.domElement);

  const ambient = new THREE.AmbientLight(0xffffff, 0.4);
  scene.add(ambient);
  scene.add(controls.getObject());

  const spark = new SparkRenderer({ renderer });
  spark.frustumCulled = false;
  scene.add(spark);
  const splatGroup = new THREE.Group();
  scene.add(splatGroup);

  state.renderer = renderer;
  state.scene = scene;
  state.camera = camera;
  state.controls = controls;
  state.spark = spark;
  state.splatGroup = splatGroup;

  window.addEventListener("resize", () => {
    if (!state.renderer || !state.camera) return;
    state.camera.aspect = window.innerWidth / window.innerHeight;
    state.camera.updateProjectionMatrix();
    state.renderer.setSize(window.innerWidth, window.innerHeight);
  });

  controls.addEventListener("lock", () => {
    state.pointerLocked = true;
    hideOverlay();
    setHUDVisible(true);
  });

  controls.addEventListener("unlock", () => {
    state.pointerLocked = false;
    showOverlay();
  });
}

async function loadSplatMesh(url) {
  const mesh = new SplatMesh({ url });
  await mesh.initialized;
  mesh.opacity = 0;
  mesh.updateVersion();
  return mesh;
}

function disposeMesh(mesh) {
  if (!mesh) return;
  try {
    mesh.parent?.remove(mesh);
  } catch (error) {
    console.warn("Failed to detach splat mesh", error);
  }
  try {
    mesh.dispose?.();
  } catch (error) {
    console.warn("Failed to dispose splat mesh", error);
  }
}

async function crossfadeToSplat(url) {
  state.fadePromise = state.fadePromise.then(async () => {
    const newMesh = await loadSplatMesh(url);
    const group = state.splatGroup ?? state.scene;
    group.add(newMesh);

    const oldMesh = state.currentMesh;
    if (!oldMesh) {
      newMesh.opacity = 1;
      newMesh.updateVersion();
      state.currentMesh = newMesh;
      return;
    }

    const start = performance.now();
    const duration = CROSSFADE_MS;

    await new Promise((resolve) => {
      function step(now) {
        const t = Math.min(1, (now - start) / duration);
        oldMesh.opacity = 1 - t;
        oldMesh.updateVersion();
        newMesh.opacity = t;
        newMesh.updateVersion();

        if (t < 1) {
          requestAnimationFrame(step);
        } else {
          resolve();
        }
      }
      requestAnimationFrame(step);
    });

    disposeMesh(oldMesh);
    newMesh.opacity = 1;
    newMesh.updateVersion();
    state.currentMesh = newMesh;
  });

  return state.fadePromise;
}

function startAnimationLoop() {
  function loop() {
    const delta = state.clock.getDelta();
    updateMovement(delta);

    if (state.renderer && state.scene && state.camera) {
      state.renderer.render(state.scene, state.camera);
    }

    updateHUD();
    requestAnimationFrame(loop);
  }
  requestAnimationFrame(loop);
}

function updateMovement(delta) {
  if (!state.controls || !state.pointerLocked) return;
  const moveDistance = MOVE_SPEED * delta;
  if (state.movement.forward) state.controls.moveForward(moveDistance);
  if (state.movement.backward) state.controls.moveForward(-moveDistance);
  if (state.movement.left) state.controls.moveRight(-moveDistance);
  if (state.movement.right) state.controls.moveRight(moveDistance);
  const object = state.controls.getObject();
  if (state.movement.up) object.position.y += moveDistance;
  if (state.movement.down) object.position.y -= moveDistance;
}

function handleKeyEvent(event, isPressed) {
  switch (event.code) {
    case "KeyW":
    case "ArrowUp":
      state.movement.forward = isPressed;
      break;
    case "KeyS":
    case "ArrowDown":
      state.movement.backward = isPressed;
      break;
    case "KeyA":
    case "ArrowLeft":
      state.movement.left = isPressed;
      break;
    case "KeyD":
    case "ArrowRight":
      state.movement.right = isPressed;
      break;
    case "Space":
      state.movement.up = isPressed;
      break;
    case "ShiftLeft":
    case "ShiftRight":
      state.movement.down = isPressed;
      break;
    default:
      break;
  }
}

function samplePose() {
  if (!state.camera) return;
  const position = state.camera.position;
  const quaternion = state.camera.quaternion;

  const pose = {
    position: [position.x, position.y, position.z],
    quaternion: [quaternion.x, quaternion.y, quaternion.z, quaternion.w],
    timestamp: performance.now() / 1000,
  };

  state.pendingPoses.push(pose);
  updateHUD();
  maybeSendPoseBatch();
}

function startPoseSampling() {
  stopPoseSampling();
  const intervalMs = Math.max(0.1, state.poseIntervalSeconds) * 1000;
  state.samplerInterval = window.setInterval(samplePose, intervalMs);
}

function stopPoseSampling() {
  if (state.samplerInterval) {
    clearInterval(state.samplerInterval);
    state.samplerInterval = null;
  }
}

function maybeSendPoseBatch() {
  if (!state.socket || state.socket.readyState !== WebSocket.OPEN) return;
  if (state.generationInFlight) return;
  if (!state.lastGenerationId) return;
  if (state.pendingPoses.length === 0) return;

  const poses = state.pendingPoses.splice(0, state.pendingPoses.length);
  const payload = {
    type: "poseBatch",
    poses,
    sinceGenerationId: state.lastGenerationId,
  };

  try {
    state.socket.send(JSON.stringify(payload));
    state.generationInFlight = true;
  } catch (error) {
    console.error("Failed to send pose batch", error);
    state.pendingPoses.unshift(...poses);
    state.generationInFlight = false;
  }

  updateHUD();
}

function handleSocketClose(event) {
  state.connectionStatus = "Disconnected";
  state.errorMessage = event.reason || null;
  state.generationInFlight = false;
  stopPoseSampling();
  updateHUD();
  showOverlay();
}

function handleSocketError(event) {
  console.error("WebSocket error", event);
  state.errorMessage = "WebSocket error";
  updateHUD();
}

async function handleBootstrap(payload) {
  state.poseIntervalSeconds = Number(payload.poseIntervalSeconds ?? 2);
  state.windowSize = payload.windowSize ?? null;
  state.lastGenerationId = payload.generationId ?? null;
  state.connectionStatus = "Ready";

  updateHUD();
  await crossfadeToSplat(payload.splatUrl);

  if (!state.samplerInterval) {
    samplePose();
    startPoseSampling();
  }
  maybeSendPoseBatch();
}

async function handleGeneration(payload) {
  state.lastGenerationId = payload.generationId ?? state.lastGenerationId;
  state.lastLatencyMs = payload.latencyMs ?? null;
  state.connectionStatus = "Streaming";
  updateHUD();

  await crossfadeToSplat(payload.splatUrl);

  state.generationInFlight = false;
  updateHUD();
  if (state.pendingPoses.length > 0) {
    maybeSendPoseBatch();
  }
}

function initWebSocket() {
  const protocol = window.location.protocol === "https:" ? "wss" : "ws";
  const url = `${protocol}://${window.location.host}/ws`;
  const socket = new WebSocket(url);

  state.socket = socket;
  state.connectionStatus = "Connecting";
  updateHUD();

  socket.addEventListener("open", () => {
    state.connectionStatus = "Connected";
    state.errorMessage = null;
    updateHUD();
  });

  socket.addEventListener("message", async (event) => {
    let payload;
    try {
      payload = JSON.parse(event.data);
    } catch (error) {
      console.warn("Received malformed message", error);
      return;
    }

    switch (payload.type) {
      case "bootstrap":
        await handleBootstrap(payload);
        break;
      case "generation":
        await handleGeneration(payload);
        break;
      case "error":
        state.errorMessage = payload.message || "Server error";
        state.generationInFlight = false;
        updateHUD();
        break;
      default:
        console.warn("Unknown message type", payload);
    }
  });

  socket.addEventListener("close", handleSocketClose);
  socket.addEventListener("error", handleSocketError);
}

function initControls() {
  window.addEventListener("keydown", (event) => {
    handleKeyEvent(event, true);
  });
  window.addEventListener("keyup", (event) => {
    handleKeyEvent(event, false);
  });

  startButton?.addEventListener("click", () => {
    state.controls?.lock();
  });

  document.body.addEventListener(
    "click",
    () => {
      if (!state.pointerLocked) {
        state.controls?.lock();
      }
    },
    { once: true },
  );
}

async function main() {
  try {
    setupScene();
    initControls();
    initWebSocket();
    startAnimationLoop();
    updateHUD();
  } catch (error) {
    console.error("Failed to initialise VR Dream frontend", error);
    state.errorMessage = error.message;
    updateHUD();
  }
}

window.addEventListener("beforeunload", () => {
  stopPoseSampling();
  if (state.socket && state.socket.readyState === WebSocket.OPEN) {
    state.socket.close(1000, "Client navigating away");
  }
});

main();
