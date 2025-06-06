import time
import argparse
from world.world import World
from ui.map_view import MapView, worker_assignment_dialog
from ui.defense_building_ui import choose_defenses
from typing import Optional
from game.game import Game
from game.buildings import Building

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the interactive Globe-Trotter demo."
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Exit without saving the game state",
    )
    args = parser.parse_args()

    # Let the player choose a settlement via the map UI
    world = World(width=20, height=20)
    view = MapView(world)
    settlement = view.run()
    game: Optional[Game] = None

    if settlement:
        q, r = settlement
        hex_data = world.get(q, r)
        print(f"Settlement chosen at ({q}, {r}) with terrain {hex_data.terrain}")

        # Let the player pick defensive structures
        buildings = choose_defenses()
        game = Game()
        game.place_initial_settlement(q, r)

        for b in buildings:
            game.build_for_player(b)

        game.begin()
    else:
        print("No settlement chosen. Exiting.")
        return

    # Main loop: advance game state every second
    try:
        while True:
            if game.player_faction and game.player_faction.manual_assignment:
                workers = worker_assignment_dialog(game.player_faction)
                assigned = game.player_faction.workers.assigned
                if workers > assigned:
                    game.faction_manager.assign_workers(
                        game.player_faction, workers - assigned
                    )
                elif workers < assigned:
                    game.faction_manager.unassign_workers(
                        game.player_faction, assigned - workers
                    )
            game.tick()
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping game...")
    finally:
        if game is not None and not args.no_save:
            game.save()
        elif args.no_save:
            print("Skipping save (--no-save)")

if __name__ == "__main__":
    main()
