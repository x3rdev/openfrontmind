// Bridge protocol: newline-delimited JSON over stdin/stdout, one game per
// process (see CLAUDE.md's parallelization section — Python spawns N of
// these as subprocesses). Every response line is prefixed with PREFIX so
// the Python side can discard anything that isn't ours; console.log is
// also redirected to stderr as a second, source-level guard, since the
// vendored engine calls console.log directly in several places and that
// writes to stdout by default in Node.
console.log = console.error;

import readline from "node:readline";
import { Difficulty, isDifficulty, Player, Team, UnitType } from "../../engine/OpenFrontIO/src/core/game/Game";
import { StampedIntent } from "../../engine/OpenFrontIO/src/core/Schemas";
import { HarnessSession, reset, step } from "./harness";

const PREFIX = "OFM:";

// Without these, an exception thrown outside handle()'s try/catch (e.g. from a
// stray promise, a timer callback) kills the process with node's default
// uncaught-exception handling - which does print a stack trace, but a handler
// here guarantees it and makes the intent explicit rather than relying on
// node's default. A signal-killed or natively-crashed process (segfault) still
// won't hit these - that failure mode has no JS-level hook at all.
process.on("uncaughtException", (err) => {
  console.error("[server.ts] uncaughtException:", err.stack ?? err);
  process.exit(1);
});
process.on("unhandledRejection", (reason) => {
  console.error("[server.ts] unhandledRejection:", reason);
  process.exit(1);
});

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

// Troop capacity scales with population/tiles owned (and difficulty for bots) and
// changes over the game - the scripted AI itself gates on troops()/maxTroops() as a
// ratio (AiAttackBehavior.ts's hasReserveRatioTroops), not raw troop count, so this
// needs to travel every tick, not just at reset.
function maxTroopsBySmallID(session: HarnessSession): Record<number, number> {
  const config = session.game.config();
  return Object.fromEntries(
    session.game.players().map((p) => [p.smallID(), config.maxTroops(p)]),
  );
}

const STRUCTURE_TYPES: Record<string, UnitType> = {
  city: UnitType.City,
  defensePost: UnitType.DefensePost,
  port: UnitType.Port,
  silo: UnitType.MissileSilo,
  sam: UnitType.SAMLauncher,
  factory: UnitType.Factory,
  warship: UnitType.Warship,
};

function structuresBySmallID(session: HarnessSession): Record<number, Record<string, number>> {
  return Object.fromEntries(
    session.game.players().map((p) => [
      p.smallID(),
      Object.fromEntries(Object.entries(STRUCTURE_TYPES).map(([k, t]) => [k, p.unitCount(t)])),
    ]),
  );
}

// The engine's own authoritative land-tile count (matches WinCheckExecution's win-
// condition denominator exactly) - cheap field reads, safe to send every tick.
function totalLand(session: HarnessSession): number {
  const map = session.game.map();
  return map.numLandTiles() - map.numTilesWithFallout();
}

// packedTileUpdates only ever covers tiles whose owner/water/fallout state has
// changed at least once (see GameImpl.recordTileUpdate's call sites) - it is not a
// full terrain dump. Without this, a client reconstructing land_mask purely from
// that delta stream starts near-empty and only fills in as territory gets fought
// over, badly undercounting total land for most of a game. Sent once at reset;
// the normal tile-update stream keeps it in sync with any later water conversions.
function landBitmap(session: HarnessSession): Uint8Array {
  const map = session.game.map();
  const bitmap = new Uint8Array(map.width() * map.height());
  for (let ref = 0; ref < bitmap.length; ref++) {
    bitmap[ref] = map.isLand(ref) ? 1 : 0;
  }
  return bitmap;
}

// Impassable tiles (mountains/walls) are isLand()===true in the engine but can never
// be owned or attacked into - GameMap.ts's setWater() explicitly refuses to convert
// them, so unlike landBitmap this is permanently fixed for a tile and only needs
// sending once at reset, no tile_updates-stream upkeep required.
function impassableBitmap(session: HarnessSession): Uint8Array {
  const map = session.game.map();
  const bitmap = new Uint8Array(map.width() * map.height());
  for (let ref = 0; ref < bitmap.length; ref++) {
    bitmap[ref] = map.isImpassable(ref) ? 1 : 0;
  }
  return bitmap;
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
    // agentClientID is a harness-generated constant added to the game synchronously
    // in reset() - this is only reachable via an engine invariant break (e.g. a
    // future refactor that stops adding the human player), never a normal outcome.
    // Throwing surfaces it as a loud cmd_success=false (see the try/catch below,
    // and bridge.py's matching check) instead of a silent agentSmallID: null that
    // would corrupt every ownership mask downstream without raising anywhere.
    if (agent === null) {
      throw new Error(`agent player not found for clientID ${session.agentClientID}`);
    }
    send({
      cmd_success: true,
      agentClientID: session.agentClientID,
      agentSmallID: agent.smallID(),
      agentName: agent.name(),
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
      maxTroops: maxTroopsBySmallID(session),
      landBitmap: toBase64(landBitmap(session)),
      impassableBitmap: toBase64(impassableBitmap(session)),
      totalLand: totalLand(session),
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
      outgoingTroops: result.outgoingTroops,
      incomingTroops: result.incomingTroops,
      unitUpdates: result.unitUpdates,
      maxTroops: maxTroopsBySmallID(session),
      totalLand: totalLand(session),
      structures: result.done ? structuresBySmallID(session) : null,
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
      send({ cmd_success: false, error: (e as Error).message, stack: (e as Error).stack });
    }
  });
});
