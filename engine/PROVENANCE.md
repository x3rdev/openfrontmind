# Provenance

Vendored from [openfrontio/OpenFrontIO](https://github.com/openfrontio/OpenFrontIO),
commit `943740de3238c1ca9b8fda24609e74b65d7365a3` (2026-07-16), `src/core/` only,
under AGPLv3 (see `LICENSE`).

`src/core/worker/` was not vendored — it's the client's Web Worker boundary
(cross-thread message passing to keep a browser tab responsive), which a
headless Node harness has no use for. Parallelism here comes from OS
subprocesses instead (see root `CLAUDE.md`).

## Patches applied on top of the vendored copy

Upstream's `src/core` has real (if narrow) reverse imports into `src/client`,
which isn't vendored. All of the following are cosmetic/display concerns or
type-only unions — none touch simulation logic:

- **`GameRunner.ts`** — removed `placeName`/`placeSpawnName` name-label-placement
  calls (client rendering cache only); `playerViewData` stays permanently empty.
- **`configuration/Config.ts`** — `Player | PlayerView` → `Player` (8 sites).
  `PlayerView` is a client-side rendering shadow reconstructed from wire diffs,
  never a real simulation object.
- **`game/GameImpl.ts`** — `renderNumber(x)` → `String(x)` (cosmetic UI
  formatting); `Unit | UnitView` → `Unit`.
- **`game/UnitGrid.ts`** — `Unit | UnitView` → `Unit` (~14 sites, purely
  type-level spatial index).
- **`execution/AttackExecution.ts`**, **`execution/TradeShipExecution.ts`**,
  **`execution/TransportShipExecution.ts`** — `renderTroops`/`renderNumber` →
  `String(...)` in UI notification messages.
- **`execution/Util.ts`** — `Game | GameView` → `Game`.
- **`game/UserSettings.ts`** — removed `graphicsOverrides()`/
  `setGraphicsOverrides()` (WebGL rendering prefs; confirmed nothing in
  `engine/core` calls either method, or `Config.userSettings()` at all).
- **`game/GameUpdateUtils.ts`** — removed `applyStateUpdate()` (consumes diffs
  on the client's receiving end; confirmed unused). Kept `diffPlayerUpdate`/
  `packAttackTroopDeltas`/the two `ATTACK_DELTA_*` constants — genuinely used
  by `PlayerImpl.ts` to *produce* diffs, unrelated to the client-only half.
- **`validations/username.ts`** — deleted entirely. Confirmed zero callers of
  `validateUsername`/`validateClanTag` anywhere in `engine/core`; the whole
  file (not just its `translateText` i18n import) was unreachable.

## Files excluded as dead code (not cosmetic patches — genuinely unused)

Found via an import-reachability trace from the two real entry points
(`GameRunner.ts`, `game/NodeGameMapLoader.ts`) rather than manual guessing:

- `AnonAnimals.ts`, `ApiSchemas.ts`, `Base64.ts`, `ClanApiSchemas.ts` — REST
  API / server-only concerns (leaderboard, clan registry, anon player naming).
  `Base64.ts` is transitively dead: its only caller was `ApiSchemas.ts`.
- `game/FetchGameMapLoader.ts`, `game/TerrainSearchMap.ts` — real browser-side
  code (confirmed still used by `src/client` upstream), orphaned specifically
  by excluding `worker/` and never vendoring `src/client`.
- `game/BinaryLoaderGameMapLoader.ts`, `pathfinding/algorithms/BFS.ts` —
  confirmed dead even in the **full** upstream `src/` tree (zero importers,
  no tests) — not something our trimming caused.

Note: this reachability trace only covered `engine/core` itself, not the test
suite (`tests/core/**`, vendored afterward in Stage 4) — `FetchGameMapLoader
.test.ts` still referenced the deleted `FetchGameMapLoader.ts` and had to be
dropped too, for the same reason (real browser-`fetch()` behavior, nothing our
`NodeGameMapLoader`-based harness will ever exercise).

## Known deferred gaps

- `Util.ts`'s `DOMPurify.sanitize()` call (wired to the `quick_chat` intent's
  chat-sanitization path, `onlyImages()`) is untested in Node — the import
  itself is side-effect-free and proven safe, but the call site has never
  actually been exercised headlessly. Not a problem for anything in steps 0/1;
  revisit if/when `quick_chat` intents are ever driven programmatically.

## Re-vendoring later

To pick up a newer upstream commit: re-copy `src/core/` (excluding `worker/`),
then re-apply the patches above — the list is short and mechanical by design,
so this should be a quick diff review against a newer commit, not a redo.
