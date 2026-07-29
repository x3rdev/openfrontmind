// tsx (used for src/harness.ts's edit-run loop) miscompiles classes that mix
// constructor-parameter properties with field initializers referencing them
// (e.g. NationNukeBehavior: `atomBombPerceivedCost = this.cost(...)` reads
// `this.game` before tsx's transpiled constructor has assigned it). A real
// bundle via esbuild.build() does not have this bug — verified by comparing
// compiled output directly. So the harness is bundled once here, then run
// as plain JS, rather than executed through tsx's on-the-fly loader.
//
// The bundle is pointed at engine/'s own tsconfig.json (not env/'s) so that
// engine/'s "resources/*"/"src/*" path aliases resolve for the whole build —
// this lets engine/ stay a verbatim, unmodified clone of upstream with zero
// patches, including the resources/QuickChat.json alias import in Schemas.ts.
import esbuild from "esbuild";

await esbuild.build({
  entryPoints: ["src/cli.ts", "src/server.ts"],
  bundle: true,
  platform: "node",
  format: "esm",
  target: "es2020",
  tsconfig: "../engine/OpenFrontIO/tsconfig.json",
  outdir: "dist",
  outExtension: { ".js": ".mjs" },
});
