import torch
from torch import Tensor, nn
from modules import Encoder, LayerNorm, LigthGCNLayer, NGCFLayer, GTLayer
from param import args

class SASRecModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.item_embeddings = nn.Embedding(args.item_size, args.hidden_size, padding_idx=0)
        self.subseq_embeddings = nn.Embedding(args.num_subseq_id, args.hidden_size)
        self.position_embeddings = nn.Embedding(args.max_seq_length, args.hidden_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.all_subseq_emb = torch.zeros(args.num_subseq_id, args.hidden_size, device=self.device)
        self.all_item_emb = torch.zeros(args.item_size, args.hidden_size, device=self.device)
        self.item_encoder = Encoder()
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        self.cross_entropy = nn.CrossEntropyLoss()
        self.apply(self.init_weights)

    def add_position_embedding(self, sequence: Tensor, item_embeddings):
        position_ids = torch.arange(sequence.size(1), dtype=torch.long, device=sequence.device)
        position_ids = position_ids.unsqueeze(0).expand_as(sequence)
        position_embeddings = self.position_embeddings(position_ids)
        sequence_emb = item_embeddings + position_embeddings
        sequence_emb = self.LayerNorm(sequence_emb)
        sequence_emb = self.dropout(sequence_emb)

        return sequence_emb

    def train_forward(self, input_id, target):
        max_logits = self.forward(input_id)
        rec_loss = self.cross_entropy(max_logits, target)
        return rec_loss

    def forward(self, input_ids: Tensor):
        candidates = self.all_item_emb
        local_intension = self.local_seq_encoding(input_ids)
        local_intension = local_intension[:,-1, :] # [B, D]
        logits = torch.matmul(local_intension, candidates.transpose(0, 1))
        return logits

    def local_seq_encoding(self, input_ids):
        item_embeddings = self.all_item_emb[input_ids] # [B,L,D]
        extended_attention_mask = self.get_transformer_mask(input_ids)
        sequence_emb = self.add_position_embedding(input_ids, item_embeddings)
        item_encoded_layers = self.item_encoder(sequence_emb, extended_attention_mask, output_all_encoded_layers=True)
        item_encoded = item_encoded_layers[-1] # [B,L,D]
        return item_encoded

    def get_transformer_mask(self, input_ids: Tensor):
        attention_mask = (input_ids > 0).long()
        extended_attention_mask: Tensor = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.int64
        max_len: int = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)

        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)  # torch.uint8
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
        subsequent_mask = subsequent_mask.long()

        if args.cuda_condition:
            subsequent_mask = subsequent_mask.cuda()

        extended_attention_mask = extended_attention_mask * subsequent_mask
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=args.initializer_range)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


class GCN(nn.Module):
    """LightGCN model."""
    def __init__(self):
        super().__init__()
        self.gcn_layers = nn.Sequential(*[LigthGCNLayer() for _ in range(args.gnn_layer)])

    def forward(self, adj, subseq_emb: Tensor, target_emb: Tensor):
        ini_emb = torch.concat([subseq_emb, target_emb], axis=0)

        layers_gcn_emb_list = [ini_emb]
        for gcn in self.gcn_layers:
            # * layers_gcn_emb_list[-1]: use the last layer's output as input
            gcn_emb = gcn(adj, layers_gcn_emb_list[-1])
            layers_gcn_emb_list.append(gcn_emb)
        sum_emb = sum(layers_gcn_emb_list) / len(layers_gcn_emb_list)

        return sum_emb[: args.num_subseq_id], sum_emb[args.num_subseq_id :]