import sys

from .cli import main


def run():
    args = sys.argv[1:]
    main(args)


if __name__ == "__main__":
    run()
