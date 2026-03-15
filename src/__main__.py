import argparse
import csv
import tqdm
from src.rag import RAG
from src.vector_index import VectorIndex


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--input", metavar="QUESTIONS FILE", required=True)

    parser.add_argument('--index-dir', metavar="INDEX DIR", required=True)
    parser.add_argument('--from-input', metavar="INPUT FILE", default=None)

    parser.add_argument("-o", "--output", metavar="ANSWERS FILE", required=True)

    args: argparse.Namespace = parser.parse_args()

    index = VectorIndex.from_args(args)

    rag = RAG(index)

    with open(args.input, "r") as f:
        reader = csv.DictReader(f)
        questions = [str(row["question"]) for row in reader]

    result = [{"question": question, "answer": rag.ask(question)} for question in tqdm.tqdm(questions)]

    with open(args.output, "w") as f:
        writer = csv.DictWriter(f, ["question", "answer"])
        writer.writeheader()
        writer.writerows(result)


if __name__ == "__main__":
    main()
