import time
from game.game import Game


def main():
    game = Game()
    # Player settlement placed at (0,0)
    game.place_initial_settlement(0, 0)
    game.begin()

    # Main loop calling tick every second
    while True:
        game.tick()
        time.sleep(1)


if __name__ == "__main__":
    main()
