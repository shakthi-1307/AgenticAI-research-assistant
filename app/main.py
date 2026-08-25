import asyncio

from .agents import research


def main():

    question = input(
        "Research question: "
    )

    print("\nResearching...\n")

    answer = asyncio.run(
        research(question)
    )

    print("\n" + answer)


if __name__ == "__main__":
    main()