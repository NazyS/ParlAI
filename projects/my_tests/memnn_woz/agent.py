# base copied from woz teacher agent


from parlai.core.teachers import DialogTeacher
from parlai.utils.io import PathManager
# from .build import build
from parlai.tasks.woz.build import build
from parlai.tasks.woz.agents import WoZTeacher
import os
import json
import copy

FILE_START = 'woz_'
FILE_END = '_en.json'


def _path(opt):
    build(opt)

    dt = opt['datatype'].split(':')[0]

    if dt == 'train':
        suffix = 'train'
    # Using matched set as valid and mismatched set as test
    elif dt == 'valid':
        suffix = 'validate'
    elif dt == 'test':
        suffix = 'test'
    else:
        raise RuntimeError('Not valid datatype.')

    data_path = os.path.join(opt['datapath'], 'WoZ', FILE_START + suffix + FILE_END)
    return data_path


class Memnn_woZTeacher(WoZTeacher):

    def setup_data(self, input_path):
        print('loading: ' + input_path)

        # new_episode = True

        with PathManager.open(input_path) as file:
            data = json.load(file)

        for dialogue in data:
            
            dialogue_length = len(dialogue)
            new_episode = [False]*dialogue_length
            new_episode[-1] = True
            new_episode = iter(new_episode)

            for line in dialogue['dialogue']:
                answer = [':'.join(turn_label) for turn_label in line['turn_label']]
                question = "What is the change in the dialogue state?"
                context = line['transcript']
                if answer:
                    # yield (context + '\n' + question, answer, None, None), new_episode
                    yield (context + '\n' + question, answer, None, None), next(new_episode)


class DefaultTeacher(Memnn_woZTeacher):
    pass