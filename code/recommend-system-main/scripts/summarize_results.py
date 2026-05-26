import argparse
import json
import os


FIELDS = [
    'tag',
    'model',
    'dataset',
    'runtime_sec',
    'valid_ndcg@10',
    'test_recall@10',
    'test_ndcg@10',
    'test_recall@50',
    'test_ndcg@50',
]


def read_jsonl(path):
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def metric(result, section, name):
    value = result.get(section, {}).get(name)
    return '' if value is None else value


def flatten(row):
    return {
        'tag': row.get('tag', ''),
        'model': row.get('model', ''),
        'dataset': row.get('dataset', ''),
        'runtime_sec': round(float(row.get('runtime_sec', 0)), 2),
        'valid_ndcg@10': metric(row, 'best_valid_result', 'ndcg@10'),
        'test_recall@10': metric(row, 'test_result', 'recall@10'),
        'test_ndcg@10': metric(row, 'test_result', 'ndcg@10'),
        'test_recall@50': metric(row, 'test_result', 'recall@50'),
        'test_ndcg@50': metric(row, 'test_result', 'ndcg@50'),
    }


def sort_rows(rows, key):
    if not key:
        return rows
    parts = key.split('_', 1)
    if len(parts) == 2:
        section, metric_name = parts
        section_name = 'test_result' if section == 'test' else 'best_valid_result'
        return sorted(
            rows,
            key=lambda row: float(metric(row, section_name, metric_name) or -1),
            reverse=True
        )
    return rows


def print_markdown(rows):
    lines = []
    lines.append('| ' + ' | '.join(FIELDS) + ' |')
    lines.append('| ' + ' | '.join(['---'] * len(FIELDS)) + ' |')
    for row in rows:
        flat = flatten(row)
        lines.append('| ' + ' | '.join(str(flat[field]) for field in FIELDS) + ' |')
    return '\n'.join(lines)


def print_csv(rows):
    lines = [','.join(FIELDS)]
    for row in rows:
        flat = flatten(row)
        lines.append(','.join(str(flat[field]) for field in FIELDS))
    return '\n'.join(lines)


def best_by_dataset(rows):
    best = {}
    for row in rows:
        dataset = row.get('dataset', '')
        score = metric(row, 'test_result', 'ndcg@10')
        if score == '':
            continue
        score = float(score)
        if dataset not in best or score > best[dataset][0]:
            best[dataset] = (score, row)
    return [best[key][1] for key in sorted(best)]


def write_final_summary(rows, output_path):
    rows = sort_rows(rows, 'test_ndcg@10'.replace('@', '@'))
    by_dataset = best_by_dataset(rows)
    content = [
        '# Experiment Summary',
        '',
        'Ranking criterion: test NDCG@10.',
        '',
        '## Best by dataset',
        '',
        print_markdown(by_dataset),
        '',
        '## All experiments sorted by test NDCG@10',
        '',
        print_markdown(rows),
        '',
    ]
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(content))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('result_file', nargs='?', default='results/experiment_results.jsonl')
    parser.add_argument('--format', choices=['markdown', 'csv'], default='markdown')
    parser.add_argument('--sort', choices=['none', 'test_ndcg@10', 'valid_ndcg@10'], default='none')
    parser.add_argument('--write-final-summary', default='', help='write markdown summary ranked by test NDCG@10')
    args = parser.parse_args()

    rows = read_jsonl(args.result_file)
    if args.sort != 'none':
        rows = sort_rows(rows, args.sort.replace('_', '_'))
    if args.write_final_summary:
        write_final_summary(read_jsonl(args.result_file), args.write_final_summary)
    if args.format == 'csv':
        print(print_csv(rows))
    else:
        print(print_markdown(rows))


if __name__ == '__main__':
    main()
