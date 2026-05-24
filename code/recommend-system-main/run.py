import argparse
import hashlib
import json
import os
import re
import shutil
import time
from logging import getLogger
import torch
from recbole.config import Config
from recbole.data import data_preparation
from recbole.utils import init_seed, init_logger, set_color, get_trainer

from utils import get_model, create_dataset


def _to_builtin(value):
    if hasattr(value, 'item'):
        return value.item()
    if isinstance(value, dict):
        return {key: _to_builtin(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(val) for val in value]
    return value


def append_result(result_file, payload):
    if not result_file:
        return
    result_dir = os.path.dirname(result_file)
    if result_dir:
        os.makedirs(result_dir, exist_ok=True)
    with open(result_file, 'a', encoding='utf-8') as f:
        f.write(json.dumps(_to_builtin(payload), ensure_ascii=False) + '\n')


def safe_tag(value):
    value = value.strip() or 'untagged'
    return re.sub(r'[^A-Za-z0-9_.-]+', '-', value)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def backup_best_checkpoint(trainer, checkpoint_dir, tag, payload):
    if not checkpoint_dir or not tag:
        return ''
    source = getattr(trainer, 'saved_model_file', '')
    if not source or not os.path.exists(source):
        return ''

    os.makedirs(checkpoint_dir, exist_ok=True)
    target = os.path.join(checkpoint_dir, f'{safe_tag(tag)}-best.pth')
    shutil.copy2(source, target)

    digest = sha256_file(target)
    with open(target + '.sha256', 'w', encoding='utf-8') as f:
        f.write(f'{digest}  {os.path.basename(target)}\n')
    meta = {
        'tag': tag,
        'source_checkpoint': source,
        'backup_checkpoint': target,
        'sha256': digest,
        'model': payload.get('model'),
        'dataset': payload.get('dataset'),
        'test_result': payload.get('test_result'),
        'best_valid_result': payload.get('best_valid_result'),
        'config_overrides': payload.get('config_overrides', {}),
    }
    with open(target + '.meta', 'w', encoding='utf-8') as f:
        json.dump(_to_builtin(meta), f, ensure_ascii=False, indent=2)
    return target


def run_single(model_name, dataset, pretrained_file='', result_file='', tag='', **kwargs):
    started_at = time.time()
    props = ['config/overall.yaml', f'config/{model_name}.yaml']

    model_class = get_model(model_name)

    # configurations initialization
    config = Config(model=model_class, dataset=dataset, config_file_list=props, config_dict=kwargs)
    init_seed(config['seed'], config['reproducibility'])
    # logger initialization
    init_logger(config)
    logger = getLogger()
    logger.info(config)

    # dataset filtering
    dataset = create_dataset(config)
    logger.info(dataset)

    # dataset splitting
    train_data, valid_data, test_data = data_preparation(config, dataset)

    # model loading and initialization
    model = model_class(config, train_data.dataset).to(config['device'])

    # Load pre-trained model
    if pretrained_file != '':
        checkpoint = torch.load(pretrained_file)
        logger.info(f'Loading from {pretrained_file}')
        model.load_state_dict(checkpoint['state_dict'], strict=False)
    logger.info(model)

    # trainer loading and initialization
    trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)

    # model training
    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, saved=True, show_progress=config['show_progress']
    )

    # model evaluation
    test_result = trainer.evaluate(test_data, load_best_model=True, show_progress=config['show_progress'])

    logger.info(set_color('best valid ', 'yellow') + f': {best_valid_result}')
    logger.info(set_color('test result', 'yellow') + f': {test_result}')

    finished_at = time.time()
    result_payload = {
        'tag': tag,
        'model': config['model'],
        'dataset': config['dataset'],
        'started_at': started_at,
        'finished_at': finished_at,
        'runtime_sec': finished_at - started_at,
        'best_valid_score': best_valid_score,
        'valid_score_bigger': config['valid_metric_bigger'],
        'best_valid_result': best_valid_result,
        'test_result': test_result,
        'saved_model_file': getattr(trainer, 'saved_model_file', ''),
        'config_overrides': kwargs
    }
    backup_path = backup_best_checkpoint(
        trainer,
        kwargs.get('checkpoint_dir', 'checkpoints'),
        tag,
        result_payload
    )
    if backup_path:
        result_payload['backup_checkpoint_file'] = backup_path
    append_result(result_file, result_payload)

    return config['model'], config['dataset'], result_payload


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


def collect_overrides(args, unknown_args):
    overrides = {}
    explicit_keys = [
        'epochs',
        'train_batch_size',
        'eval_batch_size',
        'learning_rate',
        'train_stage',
        'temperature',
        'hidden_dropout_prob',
        'attn_dropout_prob',
        'adaptor_dropout_prob',
        'stopping_step',
        'eval_step',
        'gpu_id',
        'seed',
        'show_progress',
        'checkpoint_dir',
    ]
    for key in explicit_keys:
        value = getattr(args, key)
        if value is not None:
            overrides[key] = value

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

if __name__ == '__main__':
    os.environ['DISABLE_VTUNE'] = '1'
    os.environ['NO_VTUNE'] = '1'
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', type=str, default='UniSRec', help='model name, including custom models and RecBole models')
    parser.add_argument('-d', type=str, default='CDs_and_Vinyl', help='dataset name')
    parser.add_argument('-p', type=str, default='', help='pre-trained model path')
    parser.add_argument('--tag', type=str, default='', help='experiment tag saved to result file')
    parser.add_argument('--result_file', type=str, default='results/experiment_results.jsonl', help='jsonl result file')
    parser.add_argument('--checkpoint_dir', type=str, default=None, help='fixed-name backup directory for best checkpoint')
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--train_batch_size', type=int, default=None)
    parser.add_argument('--eval_batch_size', type=int, default=None)
    parser.add_argument('--learning_rate', type=float, default=None)
    parser.add_argument('--train_stage', type=str, default=None, choices=['pretrain', 'inductive_ft', 'transductive_ft'])
    parser.add_argument('--temperature', type=float, default=None)
    parser.add_argument('--hidden_dropout_prob', type=float, default=None)
    parser.add_argument('--attn_dropout_prob', type=float, default=None)
    parser.add_argument('--adaptor_dropout_prob', type=float, default=None)
    parser.add_argument('--stopping_step', type=int, default=None)
    parser.add_argument('--eval_step', type=int, default=None)
    parser.add_argument('--gpu_id', type=int, default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--show_progress', type=parse_value, default=None)
    args, unparsed = parser.parse_known_args()

    overrides = collect_overrides(args, unparsed)

    model_name, dataset_name, results = run_single(
        args.m,
        args.d,
        pretrained_file=args.p,
        result_file=args.result_file,
        tag=args.tag,
        **overrides
    )
