import argparse
import os


def seq_len(value):
    value = value.strip()
    if not value:
        return 0
    return len(value.split(' '))


def check_file(path, limit=0):
    checked = 0
    with open(path, 'r', encoding='utf-8') as f:
        header = f.readline().rstrip('\n').split('\t')
        required = ['item_id_list:token_seq', 'rating_list:float_seq', 'time_gap_list:float_seq', 'recency_list:float_seq']
        missing = [field for field in required if field not in header]
        if missing:
            raise RuntimeError(f'{path}: missing fields {missing}')
        idx = {field: header.index(field) for field in required}
        for line_no, line in enumerate(f, start=2):
            parts = line.rstrip('\n').split('\t')
            lengths = [seq_len(parts[idx[field]]) for field in required]
            if len(set(lengths)) != 1:
                raise RuntimeError(f'{path}:{line_no}: sequence length mismatch {lengths}')
            checked += 1
            if limit and checked >= limit:
                break
    print(f'ok {path} checked={checked}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--data_path', default='dataset/processed')
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()

    base = os.path.join(args.data_path, args.dataset)
    for split in ['train', 'valid', 'test']:
        check_file(os.path.join(base, f'{args.dataset}.{split}.inter'), args.limit)
    struct_path = os.path.join(base, 'rich-item-struct.npy')
    maps_path = os.path.join(base, 'data.maps')
    if not os.path.exists(struct_path):
        raise RuntimeError(f'missing {struct_path}')
    if not os.path.exists(maps_path):
        raise RuntimeError(f'missing {maps_path}')
    print(f'ok {struct_path}')
    print(f'ok {maps_path}')


if __name__ == '__main__':
    main()
