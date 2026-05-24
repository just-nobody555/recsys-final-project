import argparse
import json
import math
import os
import sys
from collections import defaultdict

import torch
from recbole.config import Config
from recbole.data import data_preparation
from recbole.data.dataloader import FullSortEvalDataLoader
from recbole.utils import get_trainer, init_seed

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils import create_dataset, get_model


def parse_value(raw):
    lowered = raw.lower()
    if lowered in ['true', 'false']:
        return lowered == 'true'
    if lowered in ['none', 'null', '~']:
        return None
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    return raw


def collect_unknown_overrides(unknown_args):
    overrides = {}
    idx = 0
    while idx < len(unknown_args):
        key = unknown_args[idx]
        if not key.startswith('--'):
            raise ValueError(f'Unknown argument format: {key}')
        key = key[2:]
        if idx + 1 >= len(unknown_args) or unknown_args[idx + 1].startswith('--'):
            raise ValueError(f'Missing value for argument: --{key}')
        overrides[key] = parse_value(unknown_args[idx + 1])
        idx += 2
    return overrides


def id_to_token(dataset, field, value):
    return dataset.field2id_token[field][int(value)]


def update_metrics(metrics, true_item, ranklist):
    for k in [10, 50]:
        topk = ranklist[:k]
        recall_key = f'recall@{k}'
        ndcg_key = f'ndcg@{k}'
        metrics[recall_key] += 1.0 if true_item in topk else 0.0
        if true_item in topk:
            rank = topk.index(true_item) + 1
            metrics[ndcg_key] += 1.0 / math.log2(rank + 1)


@torch.no_grad()
def export_ranklist(args, overrides):
    model_class = get_model(args.model)
    props = ['config/overall.yaml', f'config/{args.model}.yaml']
    config = Config(model=model_class, dataset=args.dataset, config_file_list=props, config_dict=overrides)
    init_seed(config['seed'], config['reproducibility'])

    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)
    split_data = {'valid': valid_data, 'test': test_data}[args.split]
    if not isinstance(split_data, FullSortEvalDataLoader):
        raise ValueError('Ranklist export expects full-sort evaluation data.')

    model = model_class(config, train_data.dataset).to(config['device'])
    checkpoint = torch.load(args.checkpoint, map_location=config['device'])
    missing, unexpected = model.load_state_dict(checkpoint['state_dict'], strict=args.strict)
    if hasattr(model, 'load_other_parameter'):
        model.load_other_parameter(checkpoint.get('other_parameter'))
    model.eval()

    trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)
    trainer.model = model
    trainer.tot_item_num = split_data._dataset.item_num
    trainer.item_tensor = split_data._dataset.get_item_feature().to(config['device'])

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    metrics = defaultdict(float)
    count = 0
    user_field = split_data._dataset.uid_field
    item_field = split_data._dataset.iid_field

    with open(args.output, 'w', encoding='utf-8') as f:
        for batched_data in split_data:
            interaction, scores, positive_u, positive_i = trainer._full_sort_batch_eval(batched_data)
            top_ids = torch.topk(scores, k=args.topk, dim=1).indices.detach().cpu().tolist()
            users = interaction[user_field].detach().cpu().tolist()
            positives = positive_i.detach().cpu().tolist()

            for offset, item_ids in enumerate(top_ids):
                true_item = id_to_token(split_data._dataset, item_field, positives[offset])
                ranklist = [id_to_token(split_data._dataset, item_field, item_id) for item_id in item_ids]
                update_metrics(metrics, true_item, ranklist)
                row = {
                    'row_id': count,
                    'user_id': id_to_token(split_data._dataset, user_field, users[offset]),
                    'true_item': true_item,
                    'ranklist': ranklist,
                }
                f.write(json.dumps(row, ensure_ascii=False) + '\n')
                count += 1

    summary = {key: value / count for key, value in metrics.items()}
    summary.update({
        'model': args.model,
        'dataset': args.dataset,
        'split': args.split,
        'checkpoint': args.checkpoint,
        'ranklist_file': args.output,
        'num_rows': count,
        'missing_keys': list(missing),
        'unexpected_keys': list(unexpected),
    })
    if args.summary:
        os.makedirs(os.path.dirname(args.summary) or '.', exist_ok=True)
        with open(args.summary, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', required=True)
    parser.add_argument('-d', '--dataset', required=True)
    parser.add_argument('-p', '--checkpoint', required=True)
    parser.add_argument('--split', choices=['valid', 'test'], default='test')
    parser.add_argument('--topk', type=int, default=200)
    parser.add_argument('--output', required=True)
    parser.add_argument('--summary', default='')
    parser.add_argument('--strict', action='store_true')
    args, unknown = parser.parse_known_args()
    export_ranklist(args, collect_unknown_overrides(unknown))


if __name__ == '__main__':
    main()
