import time
from world.world import World
from ui.map_view import MapView, worker_assignment_dialog
from ui.defense_building_ui import choose_defenses
from ui.faction_creation import FactionCreationUI
from game.game import Game
from game.models import Position

def main():
    # Gather faction details from the creation UI
    creator = FactionCreationUI()
    creator.mainloop()
    if creator.result is None:
        print("No faction created. Exiting.")
        return

    # Let the player choose a settlement via the map UI
    world = World(width=20, height=20)
    view = MapView(world)
    settlement = view.run()

    if settlement:
        q, r = settlement
        hex_data = world.get(q, r)
        print(f"Settlement chosen at ({q}, {r}) with terrain {hex_data.terrain}")

        # Let the player pick defensive structures
        buildings = choose_defenses()
        game = Game()

        faction = creator.to_faction(q, r, game.world)
        if game.map.is_occupied(Position(q, r)):
            raise ValueError("Cannot place settlement on occupied location")
        game.player_faction = faction
        game.map.add_faction(faction)
        game._register_faction(faction)
        game.population = faction.citizens.count

        for b in buildings:
            game.add_building(b)

        game.begin()
    else:
        print("No settlement chosen. Exiting.")
        return

    # Main loop: advance game state every second
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

if __name__ == "__main__":
    main()
