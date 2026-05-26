import os
import re
import html
import json
import argparse
import sys
import bisect
import math
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

def load_dataset(domain, input_dir):
    train_path = os.path.join(input_dir, domain, 'train.csv.gz')
    valid_path = os.path.join(input_dir, domain, 'valid.csv.gz')
    test_path = os.path.join(input_dir, domain, 'test.csv.gz')
    
    train_df = pd.read_csv(train_path, compression='gzip')
    valid_df = pd.read_csv(valid_path, compression='gzip')
    test_df = pd.read_csv(test_path, compression='gzip')

    for df in [train_df, valid_df, test_df]:
        df['history'] = df['history'].fillna('')

    return {'train': train_df, 'valid': valid_df, 'test': test_df}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--domain', type=str, default='Musical_Instruments', choices=['Industrial_and_Scientific', 'Musical_Instruments', 'CDs_and_Vinyl'], help='domain of the dataset')
    parser.add_argument('--input_dir', type=str, default='dataset/data', help='directory containing raw datasets')
    parser.add_argument('--max_his_len', type=int, default=50, help='maximum length of the history')
    parser.add_argument('--n_workers', type=int, default=16, help='number of worker threads for parallel processing')
    parser.add_argument('--output_dir', type=str, default='dataset/processed', help='directory to save the processed datasets')
    parser.add_argument('--output_domain', type=str, default=None, help='processed dataset name, defaults to --domain')
    parser.add_argument('--device', type=str, default='mps', help='device to use')
    parser.add_argument('--plm', type=str, default='hyp1231/blair-roberta-base', help='pretrained language model to use')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--skip_plm_features', action='store_true', help='only write .inter and data.maps files')
    parser.add_argument('--rich_metadata', action='store_true', help='include additional metadata fields in PLM text')
    parser.add_argument('--with_user_features', action='store_true', help='write rating/time/recency history sequences')
    return parser.parse_args()

def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
        
def filter_items_wo_metadata_row(row, item2meta):
    if row['parent_asin'] not in item2meta:
        row['history'] = ''
    history = row['history'].split(' ')
    filtered_history = [_ for _ in history if _ in item2meta]
    row['history'] = ' '.join(filtered_history)
    return row

def truncate_history_row(row, max_his_len):
    history_items = row['history'].split(' ')
    if len(history_items) > max_his_len:
        row['history'] = ' '.join(history_items[-max_his_len:])
    return row


def remap_id(datasets):
    user2id = {'[PAD]': 0}
    id2user = ['[PAD]']
    item2id = {'[PAD]': 0}
    id2item = ['[PAD]']

    for split in ['train', 'valid', 'test']:
        dataset = datasets[split]
        for user_id, item_id, history in zip(dataset['user_id'], dataset['parent_asin'], dataset['history']):
            if user_id not in user2id:
                user2id[user_id] = len(id2user)
                id2user.append(user_id)
            if item_id not in item2id:
                item2id[item_id] = len(id2item)
                id2item.append(item_id)
            items_in_history = history.split(' ')
            for item in items_in_history:
                if item not in item2id:
                    item2id[item] = len(id2item)
                    id2item.append(item)

    data_maps = {'user2id': user2id, 'id2user': id2user, 'item2id': item2id, 'id2item': id2item}
    return data_maps


def list_to_str(l):
    if isinstance(l, list):
        return ', '.join(list_to_str(item) for item in l)
    else:
        return l


def clean_text(raw_text):
    if raw_text is None:
        return ''
    if isinstance(raw_text, float) and pd.isna(raw_text):
        return ''
    text = list_to_str(raw_text)
    text = html.unescape(text)
    text = text.strip()
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'[\n\t]', ' ', text)
    text = re.sub(r' +', ' ', text)
    text=re.sub(r'[^\x00-\x7F]', ' ', text)
    return text


def feature_process(feature):
    sentence = ""
    if isinstance(feature, float):
        sentence += str(feature)
        sentence += '.'
    elif isinstance(feature, list) and len(feature) > 0:
        for v in feature:
            sentence += clean_text(v)
            sentence += ', '
        sentence = sentence[:-2]
        sentence += '.'
    else:
        sentence = clean_text(feature)
    return sentence + ' '


def flatten_details(details):
    if not isinstance(details, dict):
        return ''
    chunks = []
    for key, value in details.items():
        if isinstance(value, dict):
            value = ', '.join(f'{k}: {v}' for k, v in value.items())
        chunks.append(f'{key}: {list_to_str(value)}')
    return '; '.join(chunks)


def parse_number(value):
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    cleaned = re.sub(r'[^0-9.]', '', str(value))
    if not cleaned:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def bucket_by_edges(value, edges):
    value = parse_number(value)
    if value is None:
        return 0
    for idx, edge in enumerate(edges, start=1):
        if value <= edge:
            return idx
    return len(edges) + 1


def log_bucket(value, max_bucket=21):
    value = parse_number(value)
    if value is None or value <= 0:
        return 0
    return min(int(math.log2(value + 1)) + 1, max_bucket)


def rating_bucket(value):
    value = parse_number(value)
    if value is None:
        return 0
    return min(max(int(round(value * 2)), 1), 10)


def best_seller_rank(details):
    if not isinstance(details, dict):
        return None
    rank = details.get('Best Sellers Rank')
    if isinstance(rank, dict):
        values = list(rank.values())
        rank = values[0] if values else None
    if isinstance(rank, list):
        rank = rank[0] if rank else None
    if rank is None:
        return None
    match = re.search(r'[\d,]+', str(rank))
    if not match:
        return None
    return parse_number(match.group(0))


def metadata_struct(row, category2id):
    categories = row.get('categories', [])
    if not isinstance(categories, list):
        categories = []
    leaf_category = clean_text(categories[-1]) if categories else ''
    if leaf_category and leaf_category not in category2id:
        category2id[leaf_category] = len(category2id) + 1

    return [
        bucket_by_edges(row.get('price'), [5, 10, 20, 50, 100, 200, 500]),
        rating_bucket(row.get('average_rating')),
        log_bucket(row.get('rating_number')),
        log_bucket(best_seller_rank(row.get('details'))),
        category2id.get(leaf_category, 0),
    ]


def clean_metadata(row, rich=False, category2id=None):
    meta_text = ''
    features_needed = ['title', 'features', 'categories', 'description']
    if rich:
        features_needed = [
            'title',
            'main_category',
            'store',
            'categories',
            'features',
            'description',
        ]
    
    for feature in features_needed:
        meta_text += feature_process(row.get(feature))
    if rich:
        details_text = flatten_details(row.get('details'))
        rank = best_seller_rank(row.get('details'))
        meta_text += feature_process(details_text)
        meta_text += f"price bucket {bucket_by_edges(row.get('price'), [5, 10, 20, 50, 100, 200, 500])}. "
        meta_text += f"average rating bucket {rating_bucket(row.get('average_rating'))}. "
        meta_text += f"rating number bucket {log_bucket(row.get('rating_number'))}. "
        meta_text += f"best seller rank bucket {log_bucket(rank)}. "
    
    row['cleaned_metadata'] = meta_text
    if rich:
        row['struct_metadata'] = metadata_struct(row, category2id)
    return row

def process_meta(args):
    metadata_path = os.path.join(args.input_dir, args.domain, 'item_metadata.jsonl.gz')
    meta_dataset = pd.read_json(metadata_path, lines=True, compression='gzip')

    category2id = {}
    meta_dataset = meta_dataset.apply(
        lambda row: clean_metadata(row, args.rich_metadata, category2id),
        axis=1
    )

    item2meta = {}
    item2struct = {}
    for row in meta_dataset.itertuples():
        parent_asin = row.parent_asin
        cleaned_metadata = row.cleaned_metadata
        item2meta[parent_asin] = cleaned_metadata
        if args.rich_metadata:
            item2struct[parent_asin] = row.struct_metadata

    struct_dims = [9, 11, 22, 22, len(category2id) + 1]
    return item2meta, item2struct, struct_dims


def history_items(value):
    if isinstance(value, float) and pd.isna(value):
        return []
    value = str(value).strip()
    if not value:
        return []
    return value.split(' ')


def build_event_index(datasets):
    event_index = {}
    for split in ['train', 'valid', 'test']:
        for row in datasets[split].itertuples():
            user_events = event_index.setdefault(row.user_id, {})
            item_events = user_events.setdefault(row.parent_asin, [])
            item_events.append((int(row.timestamp), float(row.rating)))
    for user_events in event_index.values():
        for item_events in user_events.values():
            item_events.sort(key=lambda event: event[0])
    return event_index


def lookup_history_event(event_index, user_id, item_id, current_timestamp):
    item_events = event_index.get(user_id, {}).get(item_id, [])
    if not item_events:
        return None
    timestamps = [event[0] for event in item_events]
    pos = bisect.bisect_left(timestamps, int(current_timestamp)) - 1
    if pos < 0:
        return None
    return item_events[pos]


def time_gap_bucket(current_timestamp, history_timestamp):
    if history_timestamp is None:
        return 9
    days = max((int(current_timestamp) - int(history_timestamp)) / 86400000.0, 0)
    edges = [1, 7, 30, 90, 180, 365, 730, 1460]
    for idx, edge in enumerate(edges, start=1):
        if days <= edge:
            return idx
    return 9


def user_feature_lists(row, event_index):
    items = history_items(row.history)
    ratings = []
    gaps = []
    recencies = []
    n_items = len(items)
    for idx, item_id in enumerate(items):
        event = lookup_history_event(event_index, row.user_id, item_id, row.timestamp)
        if event is None:
            hist_timestamp, hist_rating = None, 0.0
        else:
            hist_timestamp, hist_rating = event
        ratings.append(str(min(max(int(round(hist_rating)), 0), 5)))
        gaps.append(str(time_gap_bucket(row.timestamp, hist_timestamp)))
        recencies.append(str(min(n_items - idx, 50)))
    return ratings, gaps, recencies


if __name__ == '__main__':
    args = parse_args()

    datasets = load_dataset(args.domain, args.input_dir)
    item2meta, item2struct, struct_dims = process_meta(args)
    event_index = build_event_index(datasets)
    truncated_datasets = {}

    output_domain = args.output_domain or args.domain
    output_dir = os.path.join(args.output_dir, output_domain)
    check_path(output_dir)

    for split in ['train', 'valid', 'test']:
        filtered_dataset = datasets[split].apply(
            lambda row: filter_items_wo_metadata_row(row, item2meta),
            axis=1
        )
        filtered_dataset = filtered_dataset[filtered_dataset['history'] != '']

        truncated_dataset = filtered_dataset.apply(
            lambda row: truncate_history_row(row, args.max_his_len),
            axis=1
        )
        truncated_datasets[split] = truncated_dataset

        output_path = os.path.join(output_dir, f'{output_domain}.{split}.inter')
        with open(output_path, 'w') as f:
            if args.with_user_features:
                f.write('user_id:token\titem_id_list:token_seq\trating_list:float_seq\ttime_gap_list:float_seq\trecency_list:float_seq\titem_id:token\n')
            else:
                f.write('user_id:token\titem_id_list:token_seq\titem_id:token\n')
            for row in truncated_dataset.itertuples():
                if args.with_user_features:
                    ratings, gaps, recencies = user_feature_lists(row, event_index)
                    f.write(
                        f"{row.user_id}\t{row.history}\t{' '.join(ratings)}\t"
                        f"{' '.join(gaps)}\t{' '.join(recencies)}\t{row.parent_asin}\n"
                    )
                else:
                    f.write(f"{row.user_id}\t{row.history}\t{row.parent_asin}\n")

    data_maps = remap_id(truncated_datasets)
    id2meta = {0: '[PAD]'}
    for item in item2meta:
        if item not in data_maps['item2id']:
            continue
        item_id = data_maps['item2id'][item]
        id2meta[item_id] = item2meta[item]
    data_maps['id2meta'] = id2meta
    if args.rich_metadata:
        data_maps['struct_feature_dims'] = struct_dims
    output_path = os.path.join(output_dir, f'data.maps')
    with open(output_path, 'w') as f:
        json.dump(data_maps, f)

    if args.rich_metadata:
        struct_features = np.zeros((len(data_maps['item2id']), len(struct_dims)), dtype=np.int64)
        for item, item_id in data_maps['item2id'].items():
            if item == '[PAD]':
                continue
            struct_features[item_id] = item2struct.get(item, [0] * len(struct_dims))
        np.save(os.path.join(output_dir, 'rich-item-struct.npy'), struct_features)

    if args.skip_plm_features:
        print('Skipped PLM feature extraction.')
        print(f"#Users: {len(data_maps['user2id']) - 1}")
        print(f"#Items: {len(data_maps['item2id']) - 1}")
        n_interactions = {}
        for split in ['train', 'valid', 'test']:
            n_interactions[split] = len(truncated_datasets[split])
            for history in truncated_datasets[split]['history']:
                history_items = history.split(' ')
                n_interactions[split] += len(history_items)
        print(f"#Interaction in total: {sum(n_interactions.values())}")
        print(n_interactions)
        avg_his_length = 0
        for split in ['train', 'valid', 'test']:
            avg_his_length += sum([len(_.split(' ')) for _ in truncated_datasets[split]['history']])
        avg_his_length /= sum([len(truncated_datasets[split]) for split in ['train', 'valid', 'test']])
        print(f"Average history length: {avg_his_length}")
        sys.exit(0)

    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.plm)
    model = AutoModel.from_pretrained(args.plm).to(device)
    sorted_text = []
    for i in range(1, len(data_maps['item2id'])):
        sorted_text.append(data_maps['id2meta'][i])
    
    all_embeddings = []
    for pr in tqdm(range(0, len(sorted_text), args.batch_size)):
        batch = sorted_text[pr:pr + args.batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors='pt').to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        all_embeddings.append(embeddings)
    all_embeddings = np.concatenate(all_embeddings, axis=0)
    all_embeddings.tofile(os.path.join(output_dir, f'{args.plm.split("/")[-1]}.feature'))

    print(f"#Users: {len(data_maps['user2id']) - 1}")
    print(f"#Items: {len(data_maps['item2id']) - 1}")
    n_interactions = {}
    for split in ['train', 'valid', 'test']:
        n_interactions[split] = len(truncated_datasets[split])
        for history in truncated_datasets[split]['history']:
            history_items = history.split(' ')
            n_interactions[split] += len(history_items)
    print(f"#Interaction in total: {sum(n_interactions.values())}")
    print(n_interactions)
    avg_his_length = 0
    for split in ['train', 'valid', 'test']:
        avg_his_length += sum([len(_.split(' ')) for _ in truncated_datasets[split]['history']])
    avg_his_length /= sum([len(truncated_datasets[split]) for split in ['train', 'valid', 'test']])
    print(f"Average history length: {avg_his_length}")
    print(f"Average character length of metadata: {np.mean([len(_) for _ in sorted_text])}")
