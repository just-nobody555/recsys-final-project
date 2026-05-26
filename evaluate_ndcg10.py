import argparse
import json
import math
from pathlib import Path


def ndcg_at_10(path: Path) -> tuple[int, float, float]:
    count = 0
    recall = 0.0
    ndcg = 0.0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            count += 1
            top10 = row["ranklist"][:10]
            true_item = row["true_item"]
            if true_item in top10:
                recall += 1.0
                rank = top10.index(true_item) + 1
                ndcg += 1.0 / math.log2(rank + 1)
    return count, recall / count, ndcg / count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+", type=Path)
    args = parser.parse_args()

    total_count = 0
    weighted_ndcg = 0.0
    macro_ndcg_values = []

    print("file,num_rows,recall@10,ndcg@10")
    for path in args.files:
        count, recall, ndcg = ndcg_at_10(path)
        total_count += count
        weighted_ndcg += count * ndcg
        macro_ndcg_values.append(ndcg)
        print(f"{path.name},{count},{recall:.10f},{ndcg:.10f}")

    if macro_ndcg_values:
        print(f"macro_average,,,{sum(macro_ndcg_values) / len(macro_ndcg_values):.10f}")
        print(f"weighted_average,,,{weighted_ndcg / total_count:.10f}")


if __name__ == "__main__":
    main()
