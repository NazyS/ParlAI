import os
import torch

from parlai.core.torch_ranker_agent import TorchRankerAgent
from parlai.agents.bert_ranker.bi_encoder_ranker import BiEncoderRankerAgent
from .indo_bert_dictionary import IndoBertDictionaryAgent
from .helpers import download, MODEL_PATH, MODEL_FOLDER


class IndoBiEncoderRankerAgent(BiEncoderRankerAgent):
    def __init__(self, opt, shared=None):
        # download pretrained models
        download(opt['datapath'])
        self.pretrained_path = os.path.join(
            opt['datapath'], 'models', MODEL_FOLDER, MODEL_PATH
        )
        opt['pretrained_path'] = self.pretrained_path

        self.clip = -1

        TorchRankerAgent.__init__(self, opt, shared)
        # it's easier for now to use DataParallel when
        self.NULL_IDX = self.dict.pad_idx
        self.START_IDX = self.dict.start_idx
        self.END_IDX = self.dict.end_idx
        # default one does not average
        self.rank_loss = torch.nn.CrossEntropyLoss(reduce=True, size_average=True)

    @staticmethod
    def dictionary_class():
        return IndoBertDictionaryAgent
