import subprocess
import pathlib
import json
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

    def __init__(self, seed=None, map=None, bots=None, nations=None, build=True) -> None:
        self.width = None
        self.height = None
        self.seed = seed
        self.map = map
        self.bots = bots
        self.nations = nations

        # dist/server.mjs isn't rebuilt automatically otherwise, so a stale
        # build can silently keep running old harness.ts/server.ts logic.
        env_dir = REPO_ROOT / "env"
        if build:
            _ensure_built()

        self.process = subprocess.Popen(["node", str(env_dir / "dist" / "server.mjs")],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True
        )

    def reset(self, seed=None, map=None, bots=None, nations=None) -> dict:
        # Only include a field if the caller (or the constructor) actually
        # specified one - anything left as None is omitted, so harness.ts's
        # reset() applies its own default instead of it being shadowed here.
        cmd = {"cmd": "reset"}
        seed = seed if seed is not None else self.seed
        map = map if map is not None else self.map
        bots = bots if bots is not None else self.bots
        nations = nations if nations is not None else self.nations
        if seed is not None:
            cmd["seed"] = seed
        if map is not None:
            cmd["map"] = map
        if bots is not None:
            cmd["bots"] = bots
        if nations is not None:
            cmd["nations"] = nations
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
        return json.loads(line.removeprefix("OFM:"))

