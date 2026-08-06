// Minimal in-process reset/step harness around engine/ (roadmap step 1).
// No subprocess, no bridge protocol yet — just validating the API shape
// and measuring raw ticks/sec by driving engine/'s GameRunner directly.

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

// Loads any vendored testdata map by folder name (e.g. "world", "plains") -
// every folder under tests/testdata/maps/ has the same map.bin/map4x.bin/
// manifest.json layout, so one path handles all of them uniformly instead of
// going through GameMapType/NodeGameMapLoader, which only resolves maps that
// happen to have a GameMapType enum entry. Mirrors tests/util/Setup.ts.
// nations/additionalNations default to [] for manifests that don't define
// them (the synthetic maps); gameConfig.nations being a fixed number means
// the game generates procedural nations regardless, so this is safe.
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
    // Cosmetic only - GameConfig.gameMap wants a GameMapType, but real
    // terrain always comes from loadTestMap(map) below regardless of what's
    // set here (most vendored testdata maps, e.g. "plains", have no
    // GameMapType entry at all).
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
  // Every tile/unit update seen during the spawn-phase loop below gets
  // collected here so reset()'s caller gets the full picture established by
  // spawning, not just the diff from the single last tick.
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

  // Spawn phase: SpawnTimerExecution ends it once config.numSpawnPhaseTurns
  // is reached. Drive empty-intent ticks until then, same as FullGamePerf.ts.
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
  // null targetID means unclaimed land ("wilderness") rather than a player.
  targetID: PlayerID | null;
  smallID: number | null;
  // true if the agent already has an active outgoing attack toward this target -
  // re-attacking would just reinforce it (AttackExecution merges troops into the
  // existing attack), so this lets the Python side mask out the redundant reselect.
  alreadyAttacking: boolean;
}

export interface StepResult {
  tick: number;
  playersAlive: number;
  done: boolean;
  winner: Player | Team | null;
  packedTileUpdates?: Uint32Array;
  packedPlayerUpdates?: Float64Array;
  attackMask: AttackTarget[];
  unitUpdates?: UnitUpdate[];
}

// Sentinel key for wilderness in the active-attack set below - distinct from any
// real PlayerID since those come from the engine's own ID space.
const WILDERNESS_KEY = "__wilderness__";

// nearby() is what the engine's own scripted AI uses too.
function computeAttackMask(
  game: ReturnType<typeof createGame>,
  agentClientID: string,
): AttackTarget[] {
  const agent = game.playerByClientID(agentClientID);
  if (agent === null) return [];

  const activeTargets = new Set(
    agent
      .outgoingAttacks()
      .filter((a) => a.isActive())
      .map((a) => (a.target().isPlayer() ? (a.target() as Player).id() : WILDERNESS_KEY)),
  );

  return agent
    .nearby()
    .map((p) =>
      p.isPlayer()
        ? {
            targetID: p.id(),
            smallID: p.smallID(),
            alreadyAttacking: activeTargets.has(p.id()),
          }
        : {
            targetID: null,
            smallID: null,
            alreadyAttacking: activeTargets.has(WILDERNESS_KEY),
          },
    );
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
  return {
    tick: game.ticks(),
    playersAlive: game.players().filter((p) => p.isAlive()).length,
    done: winner !== null,
    winner,
    packedTileUpdates: state.packedTileUpdates,
    packedPlayerUpdates: state.packedPlayerUpdates,
    attackMask: computeAttackMask(game, session.agentClientID),
    unitUpdates: state.unitUpdates,
  };
}
