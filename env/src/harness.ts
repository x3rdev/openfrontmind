// Minimal in-process reset/step harness around engine/ (roadmap step 1).
// No subprocess, no bridge protocol yet — just validating the API shape
// and measuring raw ticks/sec by driving engine/'s GameRunner directly.

// TODO: reset(seed, map, bots) — build Config, load the map via
// NodeGameMapLoader, create nations/bots, create the game, wire up
// GameRunner + Executor, drive through the spawn phase, return the
// initial observation/state.

import path from "path";
import { fileURLToPath } from "url";
import {NodeGameMapLoader} from "../../engine/core/game/NodeGameMapLoader";


const PROJECT_ROOT = path.resolve(
  path.dirname(fileURLToPath(import.meta.url)),
  "../../engine/tests/testdata/maps",
);

export function reset() {
  const mapLoader = new NodeGameMapLoader(
    path.join(PROJECT_ROOT, "resources/maps"),
  );
  throw new Error("not implemented");
}

// TODO: step(intents) — addTurn + executeNextTick, return the new
// observation, latest state hash, players-alive count, and whether
// the episode is done.
export function step() {
  throw new Error("not implemented");
}

// TODO: main() — CLI entrypoint. Run reset() + N ticks of step() with
// empty intents (no policy yet), time it, print ticks/sec. Then run the
// identical seed a second time and compare hashes (self-consistency).
// Also worth cross-checking the hash against upstream's own
// FullGamePerf.ts run for the same seed, per the earlier discussion on
// verifying fidelity, not just internal determinism.
function main() {
  throw new Error("not implemented");
}

main();
