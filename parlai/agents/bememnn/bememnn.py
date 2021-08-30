from typing import Optional
import logging

from parlai.agents.bert_ranker.bi_encoder_ranker import BiEncoderRankerAgent
from parlai.agents.bert_ranker.helpers import surround, add_common_args
from parlai.agents.memnn.memnn import MemnnAgent
from parlai.core.torch_ranker_agent import TorchRankerAgent
from parlai.core.params import ParlaiParser
from parlai.core.opt import Opt

from .modules import BEMemNN, opt_to_kwargs_from_memnn, opt_to_bert_dict

logger = logging.getLogger(__name__)

class BememnnAgent(MemnnAgent, BiEncoderRankerAgent):

    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        # add bert ranker args
        add_common_args(parser)
        # add only essential memnn args
        arg_group = parser.add_argument_group('BEMemNN Arguments')
        arg_group.add_argument(
            '-hops', '--hops', type=int, default=3, help='number of memory hops'
        )
        arg_group.add_argument(
            '--memsize',
            type=int,
            default=32,
            help='size of memory, set to 0 for "nomemnn" model which just '
            'embeds query and candidates and picks most similar candidate',
        )
        parser.set_defaults(
            encode_candidate_vecs=True,
            # from bert ranker
            dict_maxexs=0,  # skip building dictionary
        )

        return arg_group

    def __init__(self, opt, shared=None):
        BiEncoderRankerAgent.__init__(self, opt, shared)
        self.id = 'BEMemNN'
        self.memsize = opt['memsize']
        if self.memsize < 0:
            self.memsize = 0
        self.use_time_features = False


    def build_model(self):
        memnn_kwargs = opt_to_kwargs_from_memnn(self.opt)
        bert_opt = opt_to_bert_dict(self.opt)

        return BEMemNN(bert_opt, padding_idx=self.NULL_IDX, **memnn_kwargs)

    @staticmethod
    def dictionary_class():
        return BiEncoderRankerAgent.dictionary_class()

    def build_dictionary(self):
        return TorchRankerAgent.build_dictionary(self)

    def _set_text_vec(self, obs, history, truncate):
        obs = MemnnAgent._set_text_vec(self, obs, history, truncate)

        # text_vecs for BiEncoderRanker
        # concatenate the [CLS] and [SEP] tokens
        if (
            obs is not None
            and 'text_vec' in obs
            and 'added_start_end_tokens' not in obs
        ):
            obs.force_set(
                'text_vec', surround(obs['text_vec'], self.START_IDX, self.END_IDX)
            )
            obs['added_start_end_tokens'] = True
        return obs

        # TODO look into possible _set_text_vec method wrong implementation
        # TODO look into batchify method which is overwritten by memory network, might be inapplicable
