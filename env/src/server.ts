// Bridge protocol: newline-delimited JSON over stdin/stdout, one game per process.
// Responses are prefixed with PREFIX, and console.log goes to stderr since the engine logs to stdout.
console.log = console.error;

import readline from "node:readline";
import { Difficulty, isDifficulty, Player, Team, UnitType } from "../../engine/OpenFrontIO/src/core/game/Game";
import { StampedIntent } from "../../engine/OpenFrontIO/src/core/Schemas";
import { HarnessSession, reset, step } from "./harness";

const PREFIX = "OFM:";

// Catch exceptions thrown outside handle() so the error is always printed.
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

// null for a team win or no winner yet.
function winnerSmallID(winner: Player | Team | null): number | null {
  if (winner === null || typeof winner === "string") return null;
  return winner.smallID();
}

// maxTroops changes over the game, so it is sent every tick.
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

// The engine's land-tile count, the win-condition denominator.
function totalLand(session: HarnessSession): number {
  const map = session.game.map();
  return map.numLandTiles() - map.numTilesWithFallout();
}

// Full terrain dump sent once at reset; tile updates only cover changed tiles.
function landBitmap(session: HarnessSession): Uint8Array {
  const map = session.game.map();
  const bitmap = new Uint8Array(map.width() * map.height());
  for (let ref = 0; ref < bitmap.length; ref++) {
    bitmap[ref] = map.isLand(ref) ? 1 : 0;
  }
  return bitmap;
}

// Impassable tiles (mountains, walls) are land but can't be owned; sent once at reset.
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
    // Only fields the client sent are parsed; harness.ts applies the defaults.
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
    // Unreachable unless an engine invariant breaks; throw instead of sending a null agentSmallID.
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

// Commands run one at a time in arrival order, since reset() is async and step() is sync.
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
