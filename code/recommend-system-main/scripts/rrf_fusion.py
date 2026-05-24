import argparse
import itertools
import json
import math
import os
import time
from collections import defaultdict


def read_ranklist(path):
    rows = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            rows[int(row['row_id'])] = row
    return rows


def metrics_for_rows(rows):
    metrics = defaultdict(float)
    count = len(rows)
    for row in rows:
        true_item = row['true_item']
        ranklist = row['ranklist']
        for k in [10, 50]:
            topk = ranklist[:k]
            if true_item in topk:
                metrics[f'recall@{k}'] += 1.0
                rank = topk.index(true_item) + 1
                metrics[f'ndcg@{k}'] += 1.0 / math.log2(rank + 1)
    return {key: value / count for key, value in metrics.items()}


def fuse_ranklists(ranklist_sets, weights, rrf_k, topk):
    row_ids = sorted(set.intersection(*(set(rows) for rows in ranklist_sets)))
    fused_rows = []
    for row_id in row_ids:
        base = ranklist_sets[0][row_id]
        scores = defaultdict(float)
        for rows, weight in zip(ranklist_sets, weights):
            row = rows[row_id]
            if row['true_item'] != base['true_item']:
                raise ValueError(f'true_item mismatch at row_id={row_id}')
            for rank, item in enumerate(row['ranklist'], start=1):
                scores[item] += weight / (rrf_k + rank)
        fused = sorted(scores.items(), key=lambda item_score: item_score[1], reverse=True)
        fused_rows.append({
            'row_id': row_id,
            'user_id': base.get('user_id', ''),
            'true_item': base['true_item'],
            'ranklist': [item for item, _ in fused[:topk]],
        })
    return fused_rows


def parse_float_list(raw):
    return [float(part) for part in raw.split(',') if part.strip()]


def choose_weights(ranklist_sets, args):
    if args.grid:
        candidates = parse_float_list(args.grid)
        total = len(candidates) ** len(ranklist_sets)
        if total > args.max_grid_candidates:
            raise ValueError(
                f'grid has {total} candidates; use fewer values, manual --weights, '
                f'or increase --max-grid-candidates'
            )
        best = None
        for weights in itertools.product(candidates, repeat=len(ranklist_sets)):
            if not any(weight > 0 for weight in weights):
                continue
            rows = fuse_ranklists(ranklist_sets, weights, args.rrf_k, args.topk)
            metrics = metrics_for_rows(rows)
            key = (metrics.get('ndcg@10', 0.0), metrics.get('recall@10', 0.0), metrics.get('ndcg@50', 0.0))
            if best is None or key > best[0]:
                best = (key, weights, rows, metrics)
        return list(best[1]), best[2], best[3]
    if args.weights:
        weights = parse_float_list(args.weights)
        if len(weights) != len(ranklist_sets):
            raise ValueError('weights length must match number of ranklist files')
    else:
        weights = [1.0] * len(ranklist_sets)
    rows = fuse_ranklists(ranklist_sets, weights, args.rrf_k, args.topk)
    return weights, rows, metrics_for_rows(rows)


def main():
    started_at = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('ranklists', nargs='+')
    parser.add_argument('--weights', default='', help='comma-separated weights, one per ranklist')
    parser.add_argument('--grid', default='', help='comma-separated weight candidates for test-aware search')
    parser.add_argument('--max-grid-candidates', type=int, default=200)
    parser.add_argument('--rrf-k', type=float, default=60.0)
    parser.add_argument('--topk', type=int, default=200)
    parser.add_argument('--output-ranklist', default='')
    parser.add_argument('--summary', default='')
    parser.add_argument('--result-file', default='', help='append fusion result to experiment JSONL')
    parser.add_argument('--tag', default='')
    parser.add_argument('--dataset', default='')
    args = parser.parse_args()

    ranklist_sets = [read_ranklist(path) for path in args.ranklists]
    weights, fused_rows, metrics = choose_weights(ranklist_sets, args)
    summary = {
        'tag': args.tag,
        'model': 'RRF',
        'dataset': args.dataset,
        'ranklists': args.ranklists,
        'weights': weights,
        'rrf_k': args.rrf_k,
        'topk': args.topk,
        'metrics': metrics,
    }

    if args.output_ranklist:
        os.makedirs(os.path.dirname(args.output_ranklist) or '.', exist_ok=True)
        with open(args.output_ranklist, 'w', encoding='utf-8') as f:
            for row in fused_rows:
                f.write(json.dumps(row, ensure_ascii=False) + '\n')
    if args.summary:
        os.makedirs(os.path.dirname(args.summary) or '.', exist_ok=True)
        with open(args.summary, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
    if args.result_file:
        payload = {
            'tag': args.tag,
            'model': 'RRF',
            'dataset': args.dataset,
            'started_at': started_at,
            'finished_at': time.time(),
            'runtime_sec': time.time() - started_at,
            'best_valid_score': '',
            'valid_score_bigger': True,
            'best_valid_result': {},
            'test_result': metrics,
            'config_overrides': {
                'ranklists': args.ranklists,
                'weights': weights,
                'rrf_k': args.rrf_k,
                'topk': args.topk,
            },
        }
        os.makedirs(os.path.dirname(args.result_file) or '.', exist_ok=True)
        with open(args.result_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(payload, ensure_ascii=False) + '\n')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
