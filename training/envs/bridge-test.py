from bridge import OpenFrontBridge

with OpenFrontBridge() as bridge:
    result = bridge.reset()
    print("reset:", result)


    for i in range(int(1.0e100)):
        result = bridge.step([])
        tiles = result["packedTileUpdates"]
        players = result["packedPlayerUpdates"]
        print(
            f"step {i}: tick={result['tick']} playersAlive={result['playersAlive']} "
            f"done={result['done']} winner={result['winner']} "
            f"tileUpdates={tiles.shape if tiles is not None else None} "
            f"playerUpdates={players.shape if players is not None else None}"
        )
        if result["done"]:
            print("game over, winner:", result["winner"])
            break
