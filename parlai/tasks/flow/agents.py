from parlai.core.teachers import ParlAIDialogTeacher
import copy
import os


def _path(opt):
    dt = opt['task'].split(':')[1:]

    datafile = os.path.join(opt['datapath'], '_'.join(dt) + '.txt')

    return datafile

class FlowTeacher(ParlAIDialogTeacher):
    def __init__(self, opt, shared=None):
        opt = copy.deepcopy(opt)

        # get datafile
        opt['parlaidialogteacher_datafile'] = _path(opt)

        super().__init__(opt, shared)


class DefaultTeacher(FlowTeacher):
    pass

    # def __init__(self, opt, shared=None):
    #     opt = copy.deepcopy(opt)

    #     # get datafile
    #     opt['parlaidialogteacher_datafile'] = _path(opt)

    #     super().__init__(opt, shared)