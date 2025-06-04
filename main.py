from game.world import World
from ui.map_view import MapView


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
    else:
        print("No settlement chosen. Exiting.")


if __name__ == "__main__":
    main()
