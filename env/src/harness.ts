// In-process reset/step harness around engine/.

import fs from "fs";
import path from "path";
import {fileURLToPath} from "url";
import {Config} from "../../engine/OpenFrontIO/src/core/configuration/Config";
import {GameConfig, GameStartInfo, StampedIntent} from "../../engine/OpenFrontIO/src/core/Schemas";
import {
  Difficulty,
  GameMapSize,
  GameMapType,
  GameMode,
  GameType,
  Player,
  PlayerID,
  PlayerInfo,
  PlayerType,
  Team,
  UnitType
} from "../../engine/OpenFrontIO/src/core/game/Game";
import {genTerrainFromBin, MapManifest} from "../../engine/OpenFrontIO/src/core/game/TerrainMapLoader";
import {GameUpdateType, UnitUpdate} from "../../engine/OpenFrontIO/src/core/game/GameUpdates";
import {PseudoRandom} from "../../engine/OpenFrontIO/src/core/PseudoRandom";
import {simpleHash} from "../../engine/OpenFrontIO/src/core/Util";
import {createNationsForGame} from "../../engine/OpenFrontIO/src/core/game/NationCreation";
import {createGame} from "../../engine/OpenFrontIO/src/core/game/GameImpl";
import {Executor} from "../../engine/OpenFrontIO/src/core/execution/ExecutionManager";
import {GameRunner} from "../../engine/OpenFrontIO/src/core/GameRunner";
import {run} from "node:test";


const PROJECT_ROOT = path.resolve(
  path.dirname(fileURLToPath(import.meta.url)),
  "../../engine/OpenFrontIO/tests/testdata/maps",
);

const MAX_SPAWN_TURNS = 1000;

// Scripted city purchase for the agent, enabled with OFM_AUTO_CITY=1.
const AUTO_CITY = process.env.OFM_AUTO_CITY === "1";
const AUTO_CITY_MAX = 3;
const AUTO_CITY_COOLDOWN_TICKS = 30;
const AUTO_CITY_GRID_STRIDE = 8;

// Construction deducts gold a tick after the intent, so builds must not be issued back to back.
const BUILD_GAP_TICKS = 5;

// Loads a vendored test map by folder name (tests/testdata/maps/).
async function loadTestMap(name: string) {
  const dir = path.join(PROJECT_ROOT, name);
  const manifest: MapManifest = JSON.parse(
    fs.readFileSync(path.join(dir, "manifest.json"), "utf8"),
  );
  const gameMap = await genTerrainFromBin(
    manifest.map,
    new Uint8Array(fs.readFileSync(path.join(dir, "map.bin"))),
  );
  const miniGameMap = await genTerrainFromBin(
    manifest.map4x,
    new Uint8Array(fs.readFileSync(path.join(dir, "map4x.bin"))),
  );
  return {
    gameMap,
    miniGameMap,
    nations: manifest.nations ?? [],
    additionalNations: manifest.additionalNations ?? [],
    teamGameSpawnAreas: manifest.teamGameSpawnAreas,
  };
}

export interface HarnessState {
  turnNumber: number;
  fatalError: string | undefined;
  packedTileUpdates?: Uint32Array;
  packedPlayerUpdates?: Float64Array;
  unitUpdates?: UnitUpdate[];
}

export interface HarnessSession {
  runner: GameRunner;
  game: ReturnType<typeof createGame>;
  state: HarnessState;
  agentClientID: string;
  lastCityTick: number;
  lastBuildTick: number;
  builtCounts: Record<string, number>;
}

export async function reset(
  seed: string = "seedseed",
  map: string = "mid_plains",
  bots: number = 50,
  nations: number | "default" | "disabled" = 5,
  difficulty: Difficulty = Difficulty.Easy,
): Promise<HarnessSession> {
  const gameConfig: GameConfig = {
    // Cosmetic: real terrain always comes from loadTestMap(map).
    gameMap: GameMapType.World,
    gameMapSize: GameMapSize.Normal,
    gameMode: GameMode.FFA,
    gameType: GameType.Public,
    difficulty,
    nations,
    donateGold: false,
    donateTroops: false,
    bots,
    infiniteGold: false,
    infiniteTroops: false,
    instantBuild: false,
    randomSpawn: true, // adjust later
    waterNukes: false,
  };

  const gameStart: GameStartInfo = {
    gameID: seed,
    lobbyCreatedAt: 0,
    config: gameConfig,
    players: []
  };

  const config = new Config(gameConfig, null, false);
  const terrain = await loadTestMap(map);
  const random = new PseudoRandom(simpleHash(gameStart.gameID));

  const agentClientID = "agent001";
  const agentPlayer = new PlayerInfo(
      "openfrontmind",
      PlayerType.Human,
      agentClientID,
      random.nextID()
  );

  const createdNations = createNationsForGame(
    gameStart,
    terrain.nations,
    terrain.additionalNations,
    0,
    random,
  );
  const game = createGame(
    [agentPlayer],
    createdNations,
    terrain.gameMap,
    terrain.miniGameMap,
    config,
    terrain.teamGameSpawnAreas,
  );

  const state: HarnessState = {
    turnNumber: 0,
    fatalError: undefined,
  };
  // Collects every update from the spawn phase so reset() returns the full state.
  const spawnTileUpdates: Uint32Array[] = [];
  const spawnUnitUpdates: UnitUpdate[] = [];
  const runner = new GameRunner(
    game,
    new Executor(game, gameStart.gameID, undefined),
    (gu) => {
      if ("errMsg" in gu) {
        state.fatalError = `${gu.errMsg}\n${gu.stack ?? ""}`;
        return;
      }
      state.packedTileUpdates = gu.packedTileUpdates
      state.packedPlayerUpdates = gu.packedPlayerUpdates
      state.unitUpdates = gu.updates[GameUpdateType.Unit];
      spawnTileUpdates.push(gu.packedTileUpdates);
      spawnUnitUpdates.push(...gu.updates[GameUpdateType.Unit]);
    },
  );
  runner.init();

  // Spawn phase: drive empty ticks until SpawnTimerExecution ends it.
  while (game.inSpawnPhase()) {
    if (state.turnNumber >= MAX_SPAWN_TURNS) {
      throw new Error(`spawn phase did not end after ${MAX_SPAWN_TURNS} turns`);
    }
    runner.addTurn({ turnNumber: state.turnNumber++, intents: [] });
    const ok = runner.executeNextTick();
    if (!ok || state.fatalError !== undefined) {
      throw new Error(`game errored during spawn phase:\n${state.fatalError}`);
    }
  }

  const totalLength = spawnTileUpdates.reduce((sum, arr) => sum + arr.length, 0);
  const combined = new Uint32Array(totalLength);
  let offset = 0;
  for (const arr of spawnTileUpdates) {
    combined.set(arr, offset);
    offset += arr.length;
  }
  state.packedTileUpdates = combined;
  state.unitUpdates = spawnUnitUpdates;

  return { runner, game, state, agentClientID, lastCityTick: -Infinity, lastBuildTick: -Infinity, builtCounts: {} };
}

export interface AttackTarget {
  // null targetID means unclaimed land (wilderness).
  targetID: PlayerID | null;
  smallID: number | null;
  // true if the agent already attacks this target; re-attacking merges into that attack.
  alreadyAttacking: boolean;
  // troops this player has committed to attacking the agent.
  attackingMeTroops: number;
  // troops the agent currently has committed in active attacks on this target.
  outgoingTroops: number;
}

export interface StepResult {
  tick: number;
  playersAlive: number;
  done: boolean;
  winner: Player | Team | null;
  packedTileUpdates?: Uint32Array;
  packedPlayerUpdates?: Float64Array;
  attackMask: AttackTarget[];
  // Troops committed to the agent's outgoing attacks, and enemy troops attacking it.
  outgoingTroops: number;
  incomingTroops: number;
  unitUpdates?: UnitUpdate[];
}

// Sentinel key for wilderness in the active-attack set.
const WILDERNESS_KEY = "__wilderness__";

// nearby() is what the engine's own scripted AI uses too.
function computeAttackMask(
  game: ReturnType<typeof createGame>,
  agentClientID: string,
): { attackMask: AttackTarget[];  outgoingTroops: number; incomingTroops: number } {
  const agent = game.playerByClientID(agentClientID);
  if (agent === null) return { attackMask: [], outgoingTroops: 0, incomingTroops: 0 };

  const outgoing = agent.outgoingAttacks().filter((a) => a.isActive());
  const activeTargets = new Set(
    outgoing.map((a) => (a.target().isPlayer() ? (a.target() as Player).id() : WILDERNESS_KEY)),
  );
  const outgoingTroops = outgoing.reduce((sum, a) => sum + a.troops(), 0);
  const outgoingByTarget = new Map<string, number>();
  for (const a of outgoing) {
    const key = a.target().isPlayer() ? (a.target() as Player).id() : WILDERNESS_KEY;
    outgoingByTarget.set(key, (outgoingByTarget.get(key) ?? 0) + a.troops());
  }

  const incoming = agent.incomingAttacks().filter((a) => a.isActive());
  const incomingTroops = incoming.reduce((sum, a) => sum + a.troops(), 0);
  const incomingByAttacker = new Map<PlayerID, number>();
  for (const a of incoming) {
    const attackerID = a.attacker().id();
    incomingByAttacker.set(attackerID, (incomingByAttacker.get(attackerID) ?? 0) + a.troops());
  }

  const attackMask = agent.nearby().map((p) =>
    p.isPlayer()
      ? {
          targetID: p.id(),
          smallID: p.smallID(),
          alreadyAttacking: activeTargets.has(p.id()),
          attackingMeTroops: incomingByAttacker.get(p.id()) ?? 0,
          outgoingTroops: outgoingByTarget.get(p.id()) ?? 0,
        }
      : {
          targetID: null,
          smallID: null,
          alreadyAttacking: activeTargets.has(WILDERNESS_KEY),
          attackingMeTroops: 0,
          outgoingTroops: outgoingByTarget.get(WILDERNESS_KEY) ?? 0,
        },
  );

  return { attackMask, outgoingTroops, incomingTroops };
}

// Builds a city at the buildable site nearest the centroid of the agent's territory.
function autoCityIntent(session: HarnessSession): StampedIntent | null {
  const { game } = session;
  if (game.ticks() - session.lastBuildTick < BUILD_GAP_TICKS) return null;
  const agent = game.playerByClientID(session.agentClientID);
  if (agent === null || !agent.isAlive()) return null;
  if (game.ticks() - session.lastCityTick < AUTO_CITY_COOLDOWN_TICKS) return null;
  if (agent.unitCount(UnitType.City) >= AUTO_CITY_MAX) return null;
  if (Number(agent.gold()) < Number(game.config().unitInfo(UnitType.City).cost(game, agent))) return null;

  const map = game.map();
  const owned: [number, number][] = [];
  for (let y = 0; y < map.height(); y += AUTO_CITY_GRID_STRIDE) {
    for (let x = 0; x < map.width(); x += AUTO_CITY_GRID_STRIDE) {
      if (map.ownerID(map.ref(x, y)) === agent.smallID()) owned.push([x, y]);
    }
  }
  if (owned.length === 0) return null;
  const cx = owned.reduce((a, [x]) => a + x, 0) / owned.length;
  const cy = owned.reduce((a, [, y]) => a + y, 0) / owned.length;
  owned.sort((a, b) => (a[0] - cx) ** 2 + (a[1] - cy) ** 2 - ((b[0] - cx) ** 2 + (b[1] - cy) ** 2));
  for (const [x, y] of owned.slice(0, 20)) {
    const site = agent.canBuild(UnitType.City, map.ref(x, y));
    if (site !== false) {
      session.lastCityTick = session.lastBuildTick = game.ticks();
      return { type: "build_unit", unit: UnitType.City, tile: site, clientID: session.agentClientID } as unknown as StampedIntent;
    }
  }
  return null;
}

const ACT_EVERY = Math.max(1, parseInt(process.env.OFM_ACT_EVERY ?? "1", 10));

export function step(
  session: HarnessSession,
  intents: StampedIntent[] = [],
): StepResult {
  const { runner, game, state } = session;
  const tiles: Uint32Array[] = [];
  const players: Float64Array[] = [];
  const unitUpdates: UnitUpdate[] = [];
  for (let j = 0; j < ACT_EVERY; j++) {
    let turn = j === 0 ? intents : [];
    if (AUTO_CITY) {
      const build = autoCityIntent(session);
      if (build !== null) turn = [...turn, build];
    }
    runner.addTurn({ turnNumber: state.turnNumber++, intents: turn });
    const ok = runner.executeNextTick();
    if (!ok || state.fatalError !== undefined) {
      throw new Error(`game errored during step:\n${state.fatalError}`);
    }
    if (state.packedTileUpdates) tiles.push(state.packedTileUpdates);
    if (state.packedPlayerUpdates) players.push(state.packedPlayerUpdates);
    if (state.unitUpdates) unitUpdates.push(...state.unitUpdates);
    if (game.getWinner() !== null) break;
  }
  const concat = <T extends Uint32Array | Float64Array>(ctor: new (n: number) => T, parts: T[]): T => {
    const out = new ctor(parts.reduce((n, a) => n + a.length, 0));
    let o = 0;
    for (const a of parts) { out.set(a, o); o += a.length; }
    return out;
  };
  state.packedTileUpdates = concat(Uint32Array, tiles);
  state.packedPlayerUpdates = concat(Float64Array, players);
  state.unitUpdates = unitUpdates;
  const agentNow = game.playerByClientID(session.agentClientID);
  if (agentNow !== null) {
    for (const t of [UnitType.City]) {
      const n = agentNow.unitCount(t);
      if (n > (session.builtCounts[t] ?? 0)) console.log(`[built] ${t} #${n} tick=${game.ticks()}`);
      session.builtCounts[t] = n;
    }
  }
  const winner = game.getWinner();
  const { attackMask, outgoingTroops, incomingTroops } = computeAttackMask(game, session.agentClientID);
  return {
    tick: game.ticks(),
    playersAlive: game.players().filter((p) => p.isAlive()).length,
    done: winner !== null,
    winner,
    packedTileUpdates: state.packedTileUpdates,
    packedPlayerUpdates: state.packedPlayerUpdates,
    attackMask,
    outgoingTroops,
    incomingTroops,
    unitUpdates: state.unitUpdates,
  };
}
