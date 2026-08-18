from .agents import research


def main():
    question = input("Research question: ").strip()

    if not question:
        return

    print("\nResearching...\n")

    answer = research(question)

    print(answer)


if __name__ == "__main__":
    main()
