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
  Team
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

  return { runner, game, state, agentClientID };
}

export interface AttackTarget {
  // null targetID means unclaimed land (wilderness).
  targetID: PlayerID | null;
  smallID: number | null;
  // true if the agent already attacks this target; re-attacking merges into that attack.
  alreadyAttacking: boolean;
  // troops this player has committed to attacking the agent.
  attackingMeTroops: number;
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
        }
      : {
          targetID: null,
          smallID: null,
          alreadyAttacking: activeTargets.has(WILDERNESS_KEY),
          attackingMeTroops: 0,
        },
  );

  return { attackMask, outgoingTroops, incomingTroops };
}

export function step(
  session: HarnessSession,
  intents: StampedIntent[] = [],
): StepResult {
  const { runner, game, state } = session;
  runner.addTurn({ turnNumber: state.turnNumber++, intents });
  const ok = runner.executeNextTick();
  if (!ok || state.fatalError !== undefined) {
    throw new Error(`game errored during step:\n${state.fatalError}`);
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
