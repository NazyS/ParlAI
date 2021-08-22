from torch import nn

from parlai.agents.bert_ranker.bi_encoder_ranker import to_bert_input
from parlai.agents.bert_ranker.helpers import BertWrapper
from parlai.agents.memnn.modules import MemNN, Hop

from pytorch_pretrained_bert.modeling import BertModel


def opt_to_kwargs_from_memnn(opt):
    kwargs = {}
    for k in ['memsize', 'hops', 'dropout']:
        if k in opt:
            kwargs[k] = opt[k]
    return kwargs

def opt_to_bert_dict(opt):
    dictionary = {}
    for k in ['pretrained_path', 'out_dim', 'add_transformer_layer', 'pull_from_layer', 'bert_aggregation']:
        if k in opt:
            dictionary[k] = opt[k]
    return dictionary


class BEMemNN(MemNN):
    def __init__(
        self,
        bert_opt,
        hops=1,
        memsize=32,
        dropout=0, # not impl in parlai yet
        padding_idx=0,
    ):

        nn.Module.__init__(self)

        self.hops = hops

        def embedding(opt=bert_opt):
            return BertAsEmbedding(opt, null_idx=padding_idx)        
            
        self.query_lt = embedding()
        self.in_memory_lt = embedding()
        self.out_memory_lt = embedding()
        self.answer_embedder = embedding()

        self.memory_hop = Hop(bert_opt['out_dim'])

class BertAsEmbedding(nn.Module):
    def __init__(self, opt, null_idx) -> None:
        super().__init__()

        self.null_idx = null_idx
        self.bert = BertWrapper(
            BertModel.from_pretrained(opt['pretrained_path']),
            opt['out_dim'],
            add_transformer_layer=opt['add_transformer_layer'],
            layer_pulled=opt['pull_from_layer'],
            aggregation=opt['bert_aggregation'],
        )

    def __call__(self, batch):
        in_shape = tuple(batch.shape)
        batch = batch.view(-1, in_shape[-1])

        token_idx_ctxt, segment_idx_ctxt, mask_ctxt = to_bert_input(
            batch, self.null_idx
        )

        embedding_ctxt = self.bert(
            token_idx_ctxt, segment_idx_ctxt, mask_ctxt
        ).view(*in_shape[:-1], -1)

        return embedding_ctxt
