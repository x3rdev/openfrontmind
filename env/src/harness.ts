// Minimal in-process reset/step harness around engine/ (roadmap step 1).
// No subprocess, no bridge protocol yet — just validating the API shape
// and measuring raw ticks/sec by driving engine/'s GameRunner directly.

// TODO: reset(seed, map, bots) — build Config, load the map via
// NodeGameMapLoader, create nations/bots, create the game, wire up
// GameRunner + Executor, drive through the spawn phase, return the
// initial observation/state.

import path from "path";
import { fileURLToPath } from "url";
import { parseArgs } from "node:util";
import {NodeGameMapLoader} from "../../engine/OpenFrontIO/tests/perf/fullgame/NodeGameMapLoader";
import {Config} from "../../engine/OpenFrontIO/src/core/configuration/Config";
import {GameConfig, GameStartInfo, StampedIntent} from "../../engine/OpenFrontIO/src/core/Schemas";
import {Difficulty, GameMapSize, GameMapType, GameMode, GameType, Player, Team} from "../../engine/OpenFrontIO/src/core/game/Game";
import {loadTerrainMap} from "../../engine/OpenFrontIO/src/core/game/TerrainMapLoader";
import {PseudoRandom} from "../../engine/OpenFrontIO/src/core/PseudoRandom";
import {simpleHash} from "../../engine/OpenFrontIO/src/core/Util";
import {createNationsForGame} from "../../engine/OpenFrontIO/src/core/game/NationCreation";
import {createGame} from "../../engine/OpenFrontIO/src/core/game/GameImpl";
import {Executor} from "../../engine/OpenFrontIO/src/core/execution/ExecutionManager";
import {GameRunner} from "../../engine/OpenFrontIO/src/core/GameRunner";
import {GameUpdateType, HashUpdate} from "../../engine/OpenFrontIO/src/core/game/GameUpdates";


const PROJECT_ROOT = path.resolve(
  path.dirname(fileURLToPath(import.meta.url)),
  "../../engine/OpenFrontIO/tests/testdata/maps",
);

const MAX_SPAWN_TURNS = 1000;

export interface HarnessState {
  turnNumber: number;
  lastHash: HashUpdate | undefined;
  fatalError: string | undefined;
}

export interface HarnessSession {
  runner: GameRunner;
  game: ReturnType<typeof createGame>;
  state: HarnessState;
}

export async function reset(
  seed: string = "seedseed",
  map: GameMapType = GameMapType.World,
  bots: number = 50,
): Promise<HarnessSession> {
  const gameConfig: GameConfig = {
    gameMap: map,
    gameMapSize: GameMapSize.Normal,
    gameMode: GameMode.FFA,
    gameType: GameType.Public,
    difficulty: Difficulty.Medium,
    nations: 5,
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
    players: [],
  };


  const config = new Config(gameConfig, null, false);
  const mapLoader = new NodeGameMapLoader(PROJECT_ROOT);
  const terrain = await loadTerrainMap(
      gameConfig.gameMap,
      gameConfig.gameMapSize,
      mapLoader,
  );
  const random = new PseudoRandom(simpleHash(gameStart.gameID));
  const nations = createNationsForGame(
    gameStart,
    terrain.nations,
    terrain.additionalNations,
    0,
    random,
  );
  const game = createGame(
    [],
    nations,
    terrain.gameMap,
    terrain.miniGameMap,
    config,
    terrain.teamGameSpawnAreas,
  );

  const state: HarnessState = {
    turnNumber: 0,
    lastHash: undefined,
    fatalError: undefined,
  };
  const runner = new GameRunner(
    game,
    new Executor(game, gameStart.gameID, undefined),
    (gu) => {
      if ("errMsg" in gu) {
        state.fatalError = `${gu.errMsg}\n${gu.stack ?? ""}`;
        return;
      }
      const hashes = gu.updates[GameUpdateType.Hash] as HashUpdate[];
      if (hashes.length > 0) {
        state.lastHash = hashes[hashes.length - 1];
      }
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

  return { runner, game, state };
}

export interface StepResult {
  tick: number;
  playersAlive: number;
  lastHash: HashUpdate | undefined;
  done: boolean;
  winner: Player | Team | null;
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
    lastHash: state.lastHash,
    done: winner !== null,
    winner,
  };
}


interface RunResult {
  ticksRun: number;
  elapsedMs: number;
  finalHash: number | undefined;
}

async function runGame(
  seed: string,
  map: GameMapType,
  bots: number,
  ticks: number,
): Promise<RunResult> {
  const session = await reset(seed, map, bots);
  const start = performance.now();
  let ticksRun = 0;
  for (; ticksRun < ticks; ticksRun++) {
    const result = step(session, []);
    if (result.done) break;
  }
  const elapsedMs = performance.now() - start;
  return { ticksRun, elapsedMs, finalHash: session.state.lastHash?.hash };
}

function logRun(label: string, run: RunResult): void {
  const ticksPerSec = run.ticksRun / (run.elapsedMs / 1000);
  console.log(
    `${label}: ${run.ticksRun} ticks in ${run.elapsedMs.toFixed(0)}ms ` +
      `(${ticksPerSec.toFixed(1)} ticks/sec), hash=${run.finalHash}`,
  );
}

async function main() {
  const { values } = parseArgs({
    options: {
      map: { type: "string", default: "World" },
      ticks: { type: "string", default: "200" },
      seed: { type: "string", default: "seedseed" },
      bots: { type: "string", default: "50" },
    },
  });

  const map = GameMapType[values.map as keyof typeof GameMapType];
  if (map === undefined) {
    throw new Error(`unknown --map "${values.map}"`);
  }
  const ticks = Number(values.ticks);
  const bots = Number(values.bots);
  const seed = values.seed as string;

  console.log(`map=${map} bots=${bots} seed=${seed} ticks=${ticks}`);

  // Run the identical seed twice — this is the vendoring regression check,
  // not a perf sanity check: matching hashes prove the engine is producing
  // the same simulation state both times, i.e. it's actually deterministic.
  const run1 = await runGame(seed, map, bots, ticks);
  logRun("run 1", run1);
  const run2 = await runGame(seed, map, bots, ticks);
  logRun("run 2", run2);

  if (run1.finalHash !== undefined && run1.finalHash === run2.finalHash) {
    console.log(`PASS: identical hash across repeated runs (seed=${seed})`);
  } else {
    console.error(
      `FAIL: hash mismatch across repeated runs (seed=${seed}): ` +
        `${run1.finalHash} vs ${run2.finalHash}`,
    );
    process.exitCode = 1;
  }
}

main();
