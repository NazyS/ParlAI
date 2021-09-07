import os

from parlai.utils.misc import warn_once
from parlai.utils.io import PathManager

from pytorch_pretrained_bert import BertTokenizer
from parlai.agents.bert_ranker.bert_dictionary import BertDictionaryAgent

from .helpers import VOCAB_PATH, MODEL_FOLDER, download


class IndoBertDictionaryAgent(BertDictionaryAgent):
    def __init__(self, opt):
        super().__init__(opt)
        # initialize from vocab path
        warn_once(
            "WARNING: BERT uses a Hugging Face tokenizer; ParlAI dictionary args are ignored"
        )

        download(opt["datapath"])
        vocab_path = PathManager.get_local_path(
            os.path.join(opt["datapath"], "models", MODEL_FOLDER, VOCAB_PATH)
        )
        self.tokenizer = BertTokenizer.from_pretrained(vocab_path)

        self.start_token = "[CLS]"
        self.end_token = "[SEP]"
        self.null_token = "[PAD]"
        self.start_idx = self.tokenizer.convert_tokens_to_ids(["[CLS]"])[
            0
        ]  # should be 101
        self.end_idx = self.tokenizer.convert_tokens_to_ids(["[SEP]"])[
            0
        ]  # should be 102
        self.pad_idx = self.tokenizer.convert_tokens_to_ids(["[PAD]"])[0]  # should be 0
        # set tok2ind for special tokens
        self.tok2ind[self.start_token] = self.start_idx
        self.tok2ind[self.end_token] = self.end_idx
        self.tok2ind[self.null_token] = self.pad_idx
        # set ind2tok for special tokens
        self.ind2tok[self.start_idx] = self.start_token
        self.ind2tok[self.end_idx] = self.end_token
        self.ind2tok[self.pad_idx] = self.null_token
