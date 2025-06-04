from game.world import World
from ui.map_view import MapView
from ui.defense_building_ui import choose_defenses
from game.game import Game
from game.buildings import Building


def main():
    world = World(width=20, height=20)
    view = MapView(world)
    settlement = view.run()
    if settlement:
        q, r = settlement
        hex_data = world.get(q, r)
        print(
            f"Settlement chosen at ({q}, {r}) with terrain {hex_data['terrain']}"
        )
        buildings = choose_defenses()
        game = Game()
        game.place_initial_settlement(q, r)
        for b in buildings:
            game.add_building(b)
        game.begin()
    else:
        print("No settlement chosen. Exiting.")


if __name__ == "__main__":
    main()
