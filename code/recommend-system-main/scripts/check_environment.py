import importlib

import torch


PACKAGES = [
    'recbole',
    'transformers',
    'pandas',
    'tqdm',
    'yaml',
    'sklearn',
]


def package_version(name):
    module = importlib.import_module(name)
    return getattr(module, '__version__', 'unknown')


def main():
    print(f'torch={torch.__version__}')
    print(f'cuda_available={torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'cuda_device_count={torch.cuda.device_count()}')
        print(f'cuda_device_name={torch.cuda.get_device_name(0)}')
    for package in PACKAGES:
        print(f'{package}={package_version(package)}')


if __name__ == '__main__':
    main()
