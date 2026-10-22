import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
from training.envs.bridge import _ensure_built, OpenFrontBridge


def _worker(conn, seed, map, bots, nations):
    bridge = OpenFrontBridge(seed=seed, map=map, bots=bots, nations=nations, build=False)
    while True:
        cmd, payload = conn.recv()
        if cmd == "close":
            bridge.process.terminate()
            bridge.process.wait()
            conn.close()
            return
        # report errors to the parent instead of dying and closing the pipe
        try:
            res = bridge.reset(**payload) if cmd == "reset" else bridge.step(payload)
        except Exception as e:
            res = {"cmd_success": False, "error": f"{type(e).__name__}: {e}"}
        conn.send(res)


def _check(res: dict, env_idx: int, cmd: str) -> dict:
    if not res.get("cmd_success", True):
        raise RuntimeError(f"env {env_idx} server error on {cmd}: {res.get('error')}")
    return res


class _BridgeProxy:
    # Decoding happens inside each worker process; doing it in the parent threads contends on the GIL.
    def __init__(self, conn, env_idx: int):
        self.conn = conn
        self.env_idx = env_idx

    def reset(self, **kwargs):
        self.conn.send(("reset", kwargs))
        return _check(self.conn.recv(), self.env_idx, "reset")


class VecOpenFrontBridge():

    def __init__(self, num_envs, seed=None, map=None, bots=None, nations=None):
        self.num_envs = num_envs
        _ensure_built()
        self.conns = []
        self.procs = []
        for _ in range(num_envs):
            parent_conn, child_conn = mp.Pipe()
            p = mp.Process(target=_worker, args=(child_conn, seed, map, bots, nations), daemon=True)
            p.start()
            self.conns.append(parent_conn)
            self.procs.append(p)
        self.bridges = [_BridgeProxy(c, i) for i, c in enumerate(self.conns)]
        # used only for resets; step() talks to the pipes directly
        self.executor = ThreadPoolExecutor(max_workers=num_envs)

    def reset(self, **kwargs):
        for conn in self.conns:
            conn.send(("reset", kwargs))
        return [_check(conn.recv(), i, "reset") for i, conn in enumerate(self.conns)]

    def step(self, intents_list):
        for conn, intents in zip(self.conns, intents_list):
            conn.send(("step", intents))
        return [_check(conn.recv(), i, "step") for i, conn in enumerate(self.conns)]

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.executor.shutdown(wait=False)
        for conn in self.conns:
            conn.send(("close", None))
        for p in self.procs:
            p.join()
