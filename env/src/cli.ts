import { parseArgs } from "node:util";
import { GameMapType } from "../../engine/OpenFrontIO/src/core/game/Game";
import { reset, step } from "./harness";
import {startGame} from "../../engine/OpenFrontIO/src/client/LocalPersistantStats";

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

  const session = await reset(seed, map, bots);
  const start = performance.now();
  let ticksRun = 0;
  while (ticks == -1 || ticksRun < ticks) {
    ticksRun++;
    let [stepResult] = await Promise.all([step(session, [])]);
    if (stepResult.done) break;
  }
  const elapsedMs = performance.now() - start;
  const ticksPerSec = ticksRun / (elapsedMs / 1000);
  console.log(
    `${ticksRun} ticks in ${elapsedMs.toFixed(0)}ms (${ticksPerSec.toFixed(1)} ticks/sec)`,
  );
}

main();
