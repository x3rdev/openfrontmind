// Minimal in-process reset/step harness around engine/ (roadmap step 1).
// No subprocess, no bridge protocol yet — just validating the API shape
// and measuring raw ticks/sec by driving engine/'s GameRunner directly.

// TODO: reset(seed, map, bots) — build Config, load the map via
// NodeGameMapLoader, create nations/bots, create the game, wire up
// GameRunner + Executor, drive through the spawn phase, return the
// initial observation/state.

import path from "path";
import {fileURLToPath} from "url";
import {NodeGameMapLoader} from "../../engine/OpenFrontIO/tests/perf/fullgame/NodeGameMapLoader";
import {Config} from "../../engine/OpenFrontIO/src/core/configuration/Config";
import {GameConfig, GameStartInfo, StampedIntent} from "../../engine/OpenFrontIO/src/core/Schemas";
import {
  Difficulty,
  GameMapSize,
  GameMapType,
  GameMode,
  GameType,
  Player,
  PlayerInfo,
  PlayerType,
  Team
} from "../../engine/OpenFrontIO/src/core/game/Game";
import {loadTerrainMap} from "../../engine/OpenFrontIO/src/core/game/TerrainMapLoader";
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

export interface HarnessState {
  turnNumber: number;
  fatalError: string | undefined;
  packedTileUpdates?: Uint32Array;
  packedPlayerUpdates?: Float64Array;
}

export interface HarnessSession {
  runner: GameRunner;
  game: ReturnType<typeof createGame>;
  state: HarnessState;
  agentClientID: string;
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
    difficulty: Difficulty.Impossible,
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
    players: []
  };

  const config = new Config(gameConfig, null, false);
  const mapLoader = new NodeGameMapLoader(PROJECT_ROOT);
  const terrain = await loadTerrainMap(
      gameConfig.gameMap,
      gameConfig.gameMapSize,
      mapLoader,
  );
  const random = new PseudoRandom(simpleHash(gameStart.gameID));

  const agentClientID = "agent001";
  const agentPlayer = new PlayerInfo(
      "openfrontmind",
      PlayerType.Human,
      agentClientID,
      random.nextID()
  );

  const nations = createNationsForGame(
    gameStart,
    terrain.nations,
    terrain.additionalNations,
    0,
    random,
  );
  const game = createGame(
    [agentPlayer],
    nations,
    terrain.gameMap,
    terrain.miniGameMap,
    config,
    terrain.teamGameSpawnAreas,
  );

  const state: HarnessState = {
    turnNumber: 0,
    fatalError: undefined,
};
  // Every tile update seen during the spawn-phase loop below gets collected
  // here so reset()'s caller gets the full territory picture established by
  // spawning, not just the diff from the single last tick.
  const spawnTileUpdates: Uint32Array[] = [];
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
      spawnTileUpdates.push(gu.packedTileUpdates);
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

  return { runner, game, state, agentClientID };
}

export interface StepResult {
  tick: number;
  playersAlive: number;
  done: boolean;
  winner: Player | Team | null;
  packedTileUpdates?: Uint32Array;
  packedPlayerUpdates?: Float64Array;
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
    packedPlayerUpdates: state.packedPlayerUpdates
  };
}
