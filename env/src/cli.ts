import { parseArgs } from "node:util";
import { reset, step } from "./harness";

async function main() {
  const { values } = parseArgs({
    options: {
      map: { type: "string", default: "mid_plains" },
      ticks: { type: "string", default: "200" },
      seed: { type: "string", default: "seedseed" },
      bots: { type: "string", default: "50" },
    },
  });

  const map = values.map as string;
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
