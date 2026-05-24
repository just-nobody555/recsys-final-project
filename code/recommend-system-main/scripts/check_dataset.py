import argparse
import os


DOMAINS = [
    'Industrial_and_Scientific',
    'Musical_Instruments',
    'CDs_and_Vinyl',
]


def expected_files(domain):
    return [
        os.path.join('dataset', 'processed', domain, f'{domain}.train.inter'),
        os.path.join('dataset', 'processed', domain, f'{domain}.valid.inter'),
        os.path.join('dataset', 'processed', domain, f'{domain}.test.inter'),
        os.path.join('dataset', 'processed', domain, 'data.maps'),
        os.path.join('dataset', 'processed', domain, 'blair-roberta-base.feature'),
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--domain', choices=DOMAINS, action='append')
    args = parser.parse_args()

    domains = args.domain or DOMAINS
    missing = []
    for domain in domains:
        print(f'[{domain}]')
        for path in expected_files(domain):
            if os.path.exists(path):
                size_mb = os.path.getsize(path) / 1024 / 1024
                print(f'  ok      {path} ({size_mb:.2f} MB)')
            else:
                print(f'  missing {path}')
                missing.append(path)

    if missing:
        raise SystemExit(f'Missing {len(missing)} required dataset files.')


if __name__ == '__main__':
    main()
