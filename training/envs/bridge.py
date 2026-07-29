import subprocess
import pathlib
import json
import numpy as np
import base64

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]


def _decode(b64_str, dtype):
    if b64_str is None:
        return None
    return np.frombuffer(base64.b64decode(b64_str), dtype=dtype)

class OpenFrontBridge:

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.process.terminate()
        self.process.wait()

    def __init__(self, seed=None, map=None, bots=None):
        self.width = None
        self.height = None
        self.seed = seed
        self.map = map
        self.bots = bots
        self.process = subprocess.Popen(["node", str(REPO_ROOT / "env" / "dist" / "server.mjs")],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True
        )

    def reset(self, seed=None, map=None, bots=None):
        # Only include a field if the caller (or the constructor) actually
        # specified one - anything left as None is omitted, so harness.ts's
        # reset() applies its own default instead of it being shadowed here.
        cmd = {"cmd": "reset"}
        seed = seed if seed is not None else self.seed
        map = map if map is not None else self.map
        bots = bots if bots is not None else self.bots
        if seed is not None:
            cmd["seed"] = seed
        if map is not None:
            cmd["map"] = map
        if bots is not None:
            cmd["bots"] = bots
        res = self._send(cmd)
        self.width = res["width"]
        self.height = res["height"]
        return self._decode_packed(res)

    def step(self, intents):
        res = self._send({"cmd": "step", "intents": intents})
        return self._decode_packed(res)

    def _decode_packed(self, res):
        res["packedTileUpdates"] = _decode(res.get("packedTileUpdates"), np.uint32)
        res["packedPlayerUpdates"] = _decode(res.get("packedPlayerUpdates"), np.float64)
        return res

    def _send(self, cmd:dict):
        self.process.stdin.write(json.dumps(cmd) + "\n")
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        return json.loads(line.removeprefix("OFM:"))

