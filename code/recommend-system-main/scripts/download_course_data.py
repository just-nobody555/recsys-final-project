import argparse
import gzip
import os
import shutil
import urllib.request


DOMAINS = [
    'Industrial_and_Scientific',
    'Musical_Instruments',
    'CDs_and_Vinyl',
]


BENCHMARK_BASE = (
    'https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/'
    'benchmark/5core/last_out_w_his'
)
META_BASE = (
    'https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/'
    'raw/meta_categories'
)
REVIEW_BASE = (
    'https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/'
    'raw/review_categories'
)


def expected_files(domain, include_reviews=False):
    files = [
        (f'{BENCHMARK_BASE}/{domain}.train.csv.gz', 'train.csv.gz'),
        (f'{BENCHMARK_BASE}/{domain}.valid.csv.gz', 'valid.csv.gz'),
        (f'{BENCHMARK_BASE}/{domain}.test.csv.gz', 'test.csv.gz'),
        (f'{META_BASE}/meta_{domain}.jsonl.gz', 'item_metadata.jsonl.gz'),
    ]
    if include_reviews:
        files.append((f'{REVIEW_BASE}/{domain}.jsonl.gz', 'reviews.jsonl.gz'))
    return files


def download(url, output_path):
    if os.path.exists(output_path) and os.path.getsize(output_path) > 0 and is_valid_gzip(output_path):
        print(f'skip existing {output_path}')
        return

    tmp_path = output_path + '.part'
    print(f'download {url}')
    print(f'      -> {output_path}')
    with urllib.request.urlopen(url, timeout=60) as response:
        expected_size = response.headers.get('Content-Length')
        expected_size = int(expected_size) if expected_size else None
        with open(tmp_path, 'wb') as f:
            shutil.copyfileobj(response, f, length=1024 * 1024)
    actual_size = os.path.getsize(tmp_path)
    if expected_size is not None and actual_size != expected_size:
        raise RuntimeError(
            f'incomplete download for {url}: expected {expected_size}, got {actual_size}'
        )
    if not is_valid_gzip(tmp_path):
        raise RuntimeError(f'invalid gzip download for {url}')
    os.replace(tmp_path, output_path)


def is_valid_gzip(path):
    try:
        with gzip.open(path, 'rb') as f:
            while f.read(1024 * 1024):
                pass
        return True
    except (EOFError, OSError):
        print(f'invalid gzip, will redownload: {path}')
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--domain', choices=DOMAINS, action='append', required=True)
    parser.add_argument('--output_dir', default=os.path.join('dataset', 'data'))
    parser.add_argument('--include_reviews', action='store_true')
    args = parser.parse_args()

    for domain in args.domain:
        domain_dir = os.path.join(args.output_dir, domain)
        os.makedirs(domain_dir, exist_ok=True)
        for url, filename in expected_files(domain, args.include_reviews):
            download(url, os.path.join(domain_dir, filename))


if __name__ == '__main__':
    main()
