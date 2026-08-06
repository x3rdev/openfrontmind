// Bridge protocol: newline-delimited JSON over stdin/stdout, one game per
// process (see CLAUDE.md's parallelization section — Python spawns N of
// these as subprocesses). Every response line is prefixed with PREFIX so
// the Python side can discard anything that isn't ours; console.log is
// also redirected to stderr as a second, source-level guard, since the
// vendored engine calls console.log directly in several places and that
// writes to stdout by default in Node.
console.log = console.error;

import readline from "node:readline";
import { Difficulty, isDifficulty, Player, Team } from "../../engine/OpenFrontIO/src/core/game/Game";
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

// null for a team win (Team has no smallID) or no winner yet - lets the
// Python side look up the winner's PlayerType via reset()'s playerTypes map.
function winnerSmallID(winner: Player | Team | null): number | null {
  if (winner === null || typeof winner === "string") return null;
  return winner.smallID();
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
    // Only parse/validate what the client actually sent - anything absent
    // stays undefined so harness.ts's reset() applies its own defaults,
    // rather than duplicating (and silently overriding) them here. An
    // unrecognized map name surfaces as an ENOENT from loadTestMap, caught
    // by the try/catch around handle() below.
    const map = typeof cmd.map === "string" ? cmd.map : undefined;
    const seed = typeof cmd.seed === "string" ? cmd.seed : undefined;
    const bots = typeof cmd.bots === "number" ? cmd.bots : undefined;
    const nations =
      typeof cmd.nations === "number" || cmd.nations === "default" || cmd.nations === "disabled"
        ? cmd.nations
        : undefined;
    const difficulty = isDifficulty(cmd.difficulty) ? cmd.difficulty : undefined;

    session = await reset(seed, map, bots, nations, difficulty);
    const agent = session.game.playerByClientID(session.agentClientID);
    send({
      cmd_success: true,
      agentClientID: session.agentClientID,
      agentSmallID: agent === null ? null : agent.smallID(),
      agentName: agent === null ? null : agent.name(),
      tick: session.game.ticks(),
      playersAlive: session.game.players().filter((p) => p.isAlive()).length,
      done: false,
      winner: null,
      width: session.game.map().width(),
      height: session.game.map().height(),
      packedTileUpdates: toBase64(session.state.packedTileUpdates),
      packedPlayerUpdates: toBase64(session.state.packedPlayerUpdates),
      unitUpdates: session.state.unitUpdates,
      playerTypes: Object.fromEntries(session.game.players().map((p) => [p.smallID(), p.type()])),
    });
    return;
  }

  if (cmd.cmd === "step") {
    if (session === undefined) {
      send({ cmd_success: false, error: "step called before reset" });
      return;
    }
    const intents = Array.isArray(cmd.intents)
      ? (cmd.intents as StampedIntent[])
      : [];
    const result = step(session, intents);
    send({
      cmd_success: true,
      tick: result.tick,
      playersAlive: result.playersAlive,
      done: result.done,
      winner: winnerToJSON(result.winner),
      winnerSmallID: winnerSmallID(result.winner),
      packedTileUpdates: toBase64(result.packedTileUpdates),
      packedPlayerUpdates: toBase64(result.packedPlayerUpdates),
      attackMask: result.attackMask,
      unitUpdates: result.unitUpdates,
    });
    return;
  }

  send({ cmd_success: false, error: `unknown cmd "${String(cmd.cmd)}"` });
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
      send({ cmd_success: false, error: `invalid JSON: ${(e as Error).message}` });
      return;
    }
    try {
      await handle(cmd);
    } catch (e) {
      send({ cmd_success: false, error: (e as Error).message });
    }
  });
});
