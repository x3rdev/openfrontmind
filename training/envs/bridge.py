import subprocess
import pathlib
import json
import sys
import threading
from collections import deque
import numpy as np
import base64

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]


def _decode(b64_str: str | None, dtype) -> np.ndarray | None:
    if b64_str is None:
        return None
    return np.frombuffer(base64.b64decode(b64_str), dtype=dtype)


def _decode_packed(res: dict) -> dict:
    res["packedTileUpdates"] = _decode(res.get("packedTileUpdates"), np.uint32)
    res["packedPlayerUpdates"] = _decode(res.get("packedPlayerUpdates"), np.float64)
    res["landBitmap"] = _decode(res.get("landBitmap"), np.uint8)
    res["impassableBitmap"] = _decode(res.get("impassableBitmap"), np.uint8)
    return res

_built = False
def _ensure_built():
    global _built
    if _built:
        return
    env_dir = REPO_ROOT / "env"
    build = subprocess.run(["node", "build.mjs"], cwd=env_dir, capture_output=True, text=True)
    if build.returncode != 0:
        raise RuntimeError(f"env/ build failed:\n{build.stdout}{build.stderr}")
    _built = True

class OpenFrontBridge:

    def __enter__(self) -> "OpenFrontBridge":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.process.terminate()
        self.process.wait()

    def __init__(self, seed=None, map=None, bots=None, nations=None, difficulty=None, build=True) -> None:
        self.width = None
        self.height = None
        self.seed = seed
        self.map = map
        self.bots = bots
        self.nations = nations
        self.difficulty = difficulty

        # rebuild so a stale dist/server.mjs can't run old harness logic
        env_dir = REPO_ROOT / "env"
        if build:
            _ensure_built()

        self.process = subprocess.Popen(["node", str(env_dir / "dist" / "server.mjs")],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        # capture stderr so a crashed env server's message ends up in the raised error
        self._stderr_tail = deque(maxlen=200)
        self._stderr_thread = threading.Thread(target=self._drain_stderr, daemon=True)
        self._stderr_thread.start()

    def _drain_stderr(self) -> None:
        for line in self.process.stderr:
            self._stderr_tail.append(line.rstrip("\n"))
            print(f"[node stderr] {line}", end="", file=sys.stderr)

    def reset(self, seed=None, map=None, bots=None, nations=None, difficulty=None) -> dict:
        # fields left as None are omitted so harness.ts applies its own defaults
        cmd = {"cmd": "reset"}
        seed = seed if seed is not None else self.seed
        map = map if map is not None else self.map
        bots = bots if bots is not None else self.bots
        nations = nations if nations is not None else self.nations
        difficulty = difficulty if difficulty is not None else self.difficulty
        if seed is not None:
            cmd["seed"] = seed
        if map is not None:
            cmd["map"] = map
        if bots is not None:
            cmd["bots"] = bots
        if nations is not None:
            cmd["nations"] = nations
        if difficulty is not None:
            cmd["difficulty"] = difficulty
        res = self._send(cmd)
        self.width = res["width"]
        self.height = res["height"]
        return _decode_packed(res)

    def step(self, intents) -> dict:
        res = self._send({"cmd": "step", "intents": intents})
        return _decode_packed(res)

    def _send(self, cmd: dict) -> dict:
        self.process.stdin.write(json.dumps(cmd) + "\n")
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if line == "":
            # EOF: the node process died
            returncode = self.process.wait(timeout=5)
            stderr_tail = "\n".join(self._stderr_tail) or "(no stderr captured)"
            raise RuntimeError(
                f"env server process died during {cmd['cmd']} (exit code {returncode}):\n{stderr_tail}"
            )
        res = json.loads(line.removeprefix("OFM:"))
        # server.ts reports exceptions as cmd_success=False
        if not res.get("cmd_success", True):
            raise RuntimeError(f"env server error on {cmd['cmd']}: {res.get('error')}\n{res.get('stack', '')}")
        return res

