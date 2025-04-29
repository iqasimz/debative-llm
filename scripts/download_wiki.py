from datasets import load_dataset

def main():
    wiki = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    with open("../data/wiki.txt", "w", encoding="utf-8") as f:
        for line in wiki["text"]:
            f.write(line + "\n")

if __name__ == "__main__":
    main()