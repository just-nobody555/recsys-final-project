import torch
import torch.nn as nn
import torch.nn.functional as F

from model.unisrec import UniSRec


class RichUniSRec(UniSRec):
    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        self.use_rich_item_struct = config['use_rich_item_struct']
        self.use_user_history_features = config['use_user_history_features']
        self.rich_struct_embedding_size = config['rich_struct_embedding_size']
        self.rich_user_feature_embedding_size = config['rich_user_feature_embedding_size']
        self.rich_time_gap_bucket_num = config['rich_time_gap_bucket_num']
        self.rich_rating_bucket_num = config['rich_rating_bucket_num']
        self.rich_recency_bucket_num = config['rich_recency_bucket_num']

        if self.use_rich_item_struct:
            struct_dims = getattr(dataset, 'struct_feature_dims', [1])
            self.register_buffer('item_struct_features', dataset.struct_feature.clone().long())
            self.struct_embeddings = nn.ModuleList([
                nn.Embedding(int(dim), self.rich_struct_embedding_size, padding_idx=0)
                for dim in struct_dims
            ])
            self.struct_projection = nn.Linear(
                len(struct_dims) * self.rich_struct_embedding_size,
                self.hidden_size
            )
            self.struct_embeddings.apply(self._init_weights)
            self.struct_projection.apply(self._init_weights)

        if self.use_user_history_features:
            self.rating_embedding = nn.Embedding(
                self.rich_rating_bucket_num,
                self.rich_user_feature_embedding_size,
                padding_idx=0
            )
            self.time_gap_embedding = nn.Embedding(
                self.rich_time_gap_bucket_num,
                self.rich_user_feature_embedding_size,
                padding_idx=0
            )
            self.recency_embedding = nn.Embedding(
                self.rich_recency_bucket_num,
                self.rich_user_feature_embedding_size,
                padding_idx=0
            )
            self.user_feature_projection = nn.Linear(
                3 * self.rich_user_feature_embedding_size,
                self.hidden_size
            )
            self.rating_embedding.apply(self._init_weights)
            self.time_gap_embedding.apply(self._init_weights)
            self.recency_embedding.apply(self._init_weights)
            self.user_feature_projection.apply(self._init_weights)

    def structured_item_embedding(self, item_ids):
        if not self.use_rich_item_struct:
            return 0
        features = self.item_struct_features[item_ids]
        embeds = []
        for idx, emb_layer in enumerate(self.struct_embeddings):
            feature = features[..., idx].clamp(min=0, max=emb_layer.num_embeddings - 1)
            embeds.append(emb_layer(feature))
        return self.struct_projection(torch.cat(embeds, dim=-1))

    def user_history_embedding(self, interaction, item_seq):
        if not self.use_user_history_features:
            return 0
        try:
            rating_list = interaction['rating_list']
            time_gap_list = interaction['time_gap_list']
            recency_list = interaction['recency_list']
        except KeyError:
            return 0

        ratings = rating_list.long().clamp(
            min=0, max=self.rich_rating_bucket_num - 1
        )
        gaps = time_gap_list.long().clamp(
            min=0, max=self.rich_time_gap_bucket_num - 1
        )
        recencies = recency_list.long().clamp(
            min=0, max=self.rich_recency_bucket_num - 1
        )
        feature_emb = torch.cat([
            self.rating_embedding(ratings),
            self.time_gap_embedding(gaps),
            self.recency_embedding(recencies),
        ], dim=-1)
        feature_emb = self.user_feature_projection(feature_emb)
        return feature_emb * (item_seq > 0).unsqueeze(-1)

    def sequence_item_embedding(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_emb_list = self.moe_adaptor(self.plm_embedding(item_seq))
        item_emb_list = item_emb_list + self.structured_item_embedding(item_seq)
        item_emb_list = item_emb_list + self.user_history_embedding(interaction, item_seq)
        return item_emb_list

    def candidate_item_embedding(self):
        item_ids = torch.arange(
            self.plm_embedding.weight.size(0),
            device=self.plm_embedding.weight.device
        )
        test_item_emb = self.moe_adaptor(self.plm_embedding.weight)
        test_item_emb = test_item_emb + self.structured_item_embedding(item_ids)
        if self.train_stage == 'transductive_ft':
            test_item_emb = test_item_emb + self.item_embedding.weight
        return test_item_emb

    def calculate_loss(self, interaction):
        if self.train_stage == 'pretrain':
            return super().calculate_loss(interaction)

        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        item_emb_list = self.sequence_item_embedding(interaction)
        seq_output = self.forward(item_seq, item_emb_list, item_seq_len)
        test_item_emb = self.candidate_item_embedding()

        seq_output = F.normalize(seq_output, dim=1)
        test_item_emb = F.normalize(test_item_emb, dim=1)

        logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1)) / self.temperature
        pos_items = interaction[self.POS_ITEM_ID]
        loss = self.loss_fct(logits, pos_items)
        return loss

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        item_emb_list = self.sequence_item_embedding(interaction)
        seq_output = self.forward(item_seq, item_emb_list, item_seq_len)
        test_items_emb = self.candidate_item_embedding()

        seq_output = F.normalize(seq_output, dim=-1)
        test_items_emb = F.normalize(test_items_emb, dim=-1)

        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))
        return scores
