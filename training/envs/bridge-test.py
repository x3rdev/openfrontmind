import argparse
import time

import numpy as np

from bridge import OpenFrontBridge
from map_view import apply_tile_updates, apply_unit_updates, draw_units, outline_agent_territory, render


def main(visualize: bool, seed: str, scale: int) -> None:
    with OpenFrontBridge() as bridge:
        result = bridge.reset(seed=seed, bots=0, nations=1)
        print("reset:", result)
        agent_client_id = result["agentClientID"]
        agent_small_id = result["agentSmallID"]

        if visualize:
            import cv2

            cv2.namedWindow("map")  # user-resizable window
            width, height = result["width"], result["height"]
            owner_grid = np.zeros((height, width), dtype=np.uint32)
            land_mask = np.zeros((height, width), dtype=bool)
            units: dict = {}
            apply_tile_updates(owner_grid, land_mask, width, result["packedTileUpdates"])
            apply_unit_updates(units, result["unitUpdates"])
            img = render(owner_grid, land_mask, agent_small_id).to_bgr_uint8(scale=scale)
            draw_units(img, units, width, scale)
            cv2.imshow("map", outline_agent_territory(img, owner_grid, agent_small_id, scale))
            cv2.waitKey(1)

        # TEMPORARY: throttle so only one attack is in flight at a time,
        # instead of spamming a new one every tick - revert if not needed.
        ATTACK_COOLDOWN_TICKS = 30
        next_attack_at = 0

        attack_mask = []  # nothing to attack yet - reset()'s response has no attackMask
        start = time.perf_counter()
        try:
            for i in range(int(1.0e100)):
                intents = []
                if attack_mask and i >= next_attack_at:
                    target = attack_mask[0]
                    intents.append({
                        "type": "attack",
                        "targetID": target["targetID"],
                        "troops": None,
                        "clientID": agent_client_id,
                    })
                    next_attack_at = i + ATTACK_COOLDOWN_TICKS

                result = bridge.step(intents)
                attack_mask = result["attackMask"]
                tiles = result["packedTileUpdates"]
                players = result["packedPlayerUpdates"]
                print(
                    f"step {i}: tick={result['tick']} playersAlive={result['playersAlive']} "
                    f"done={result['done']} winner={result['winner']} "
                    f"attackMask={len(result['attackMask'])} "
                    f"tileUpdates={tiles.shape if tiles is not None else None} "
                    f"playerUpdates={players.shape if players is not None else None}"
                )

                if visualize:
                    apply_tile_updates(owner_grid, land_mask, width, tiles)
                    apply_unit_updates(units, result["unitUpdates"])
                    if i % 10 == 0:
                        img = render(owner_grid, land_mask, agent_small_id).to_bgr_uint8(scale=scale)
                        draw_units(img, units, width, scale)
                        cv2.imshow("map", outline_agent_territory(img, owner_grid, agent_small_id, scale))
                        # waitKey also pumps the window's event loop - required
                        # for it to actually repaint, not just a keypress wait.
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break

                if result["done"]:
                    print("game over, winner:", result["winner"])
                    break

            # Captured before the blocking cv2.waitKey(0) below - otherwise
            # however long the final frame stays open counts as sim time.
            elapsed = time.perf_counter() - start
        finally:
            if visualize:
                img = render(owner_grid, land_mask, agent_small_id).to_bgr_uint8(scale=scale)
                draw_units(img, units, width, scale)
                cv2.imshow("map", outline_agent_territory(img, owner_grid, agent_small_id, scale))
                cv2.waitKey(0)
                cv2.destroyAllWindows()

        steps_run = i + 1
        print(
            f"{steps_run} steps in {elapsed:.2f}s "
            f"({steps_run / elapsed:.1f} steps/sec)"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--visualize", "-v", action="store_true")
    parser.add_argument("--seed", default="seedseed")
    parser.add_argument("--scale", type=int, default=4)
    args = parser.parse_args()
    main(visualize=args.visualize, seed=args.seed, scale=args.scale)
