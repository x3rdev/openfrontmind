// Bridge protocol: newline-delimited JSON over stdin/stdout, one game per
// process (see CLAUDE.md's parallelization section — Python spawns N of
// these as subprocesses). Every response line is prefixed with PREFIX so
// the Python side can discard anything that isn't ours; console.log is
// also redirected to stderr as a second, source-level guard, since the
// vendored engine calls console.log directly in several places and that
// writes to stdout by default in Node.
console.log = console.error;

import readline from "node:readline";
import { GameMapType, Player, Team } from "../../engine/OpenFrontIO/src/core/game/Game";
import { StampedIntent } from "../../engine/OpenFrontIO/src/core/Schemas";
import { HarnessSession, reset, step } from "./harness";

const PREFIX = "OFM:";

function send(obj: unknown): void {
  process.stdout.write(PREFIX + JSON.stringify(obj) + "\n");
}

function winnerToJSON(winner: Player | Team | null): string | null {
  if (winner === null) return null;
  return typeof winner === "string" ? winner : winner.name();
}

function toBase64(
  arr: { buffer: ArrayBufferLike; byteOffset: number; byteLength: number } | undefined,
): string | null {
  if (arr === undefined) return null;
  return Buffer.from(arr.buffer, arr.byteOffset, arr.byteLength).toString("base64");
}

let session: HarnessSession | undefined;

async function handle(cmd: Record<string, unknown>): Promise<void> {
  if (cmd.cmd === "reset") {
    const mapArg = typeof cmd.map === "string" ? cmd.map : "World";
    const map = GameMapType[mapArg as keyof typeof GameMapType];
    if (map === undefined) {
      send({ ok: false, error: `unknown map "${mapArg}"` });
      return;
    }
    const seed = typeof cmd.seed === "string" ? cmd.seed : "seedseed";
    const bots = typeof cmd.bots === "number" ? cmd.bots : 50;

    session = await reset(seed, map, bots);
    send({
      ok: true,
      agentClientID: session.agentClientID,
      tick: session.game.ticks(),
      playersAlive: session.game.players().filter((p) => p.isAlive()).length,
      done: false,
      winner: null,
    });
    return;
  }

  if (cmd.cmd === "step") {
    if (session === undefined) {
      send({ ok: false, error: "step called before reset" });
      return;
    }
    const intents = Array.isArray(cmd.intents)
      ? (cmd.intents as StampedIntent[])
      : [];
    const result = step(session, intents);
    send({
      ok: true,
      tick: result.tick,
      playersAlive: result.playersAlive,
      done: result.done,
      winner: winnerToJSON(result.winner),
      packedTileUpdates: toBase64(result.packedTileUpdates),
      packedPlayerUpdates: toBase64(result.packedPlayerUpdates),
    });
    return;
  }

  send({ ok: false, error: `unknown cmd "${String(cmd.cmd)}"` });
}

// Commands must be handled strictly one-at-a-time, in arrival order — reset()
// is async (loads the map) while step() is sync, so without this queue a
// step sent right after reset could race ahead and reply first.
let queue: Promise<void> = Promise.resolve();

const rl = readline.createInterface({ input: process.stdin });
rl.on("line", (line) => {
  if (line.trim() === "") return;
  queue = queue.then(async () => {
    let cmd: Record<string, unknown>;
    try {
      cmd = JSON.parse(line);
    } catch (e) {
      send({ ok: false, error: `invalid JSON: ${(e as Error).message}` });
      return;
    }
    try {
      await handle(cmd);
    } catch (e) {
      send({ ok: false, error: (e as Error).message });
    }
  });
});
