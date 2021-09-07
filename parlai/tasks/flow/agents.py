from parlai.core.teachers import ParlAIDialogTeacher, FbDeprecatedDialogTeacher
import copy
import os


def _path(opt):
    dt = opt['datatype'].split(':')[0]
    # if dt == 'train':
    #     suffix = 'train'
    # elif dt == 'valid':
    #     suffix = 'valid'

    tasks = opt['task'].split(':')

    datafile = os.path.join(
        opt['datapath'],
        tasks[0],
        '_'.join([
            tasks[1], dt, tasks[2]
        ]) + '.txt'
    )

    cands_datafile = os.path.join(
        opt['datapath'],
        tasks[0],
        'candidates.txt'
    )

    return datafile, cands_datafile

# class FlowTeacher(ParlAIDialogTeacher):
class FlowTeacher(FbDeprecatedDialogTeacher):
    def __init__(self, opt, shared=None):
        opt = copy.deepcopy(opt)

        # get datafile
        # opt['parlaidialogteacher_datafile'], opt['cands_datafile'] = _path(opt)
        opt['datafile'], opt['cands_datafile'] = _path(opt)

        super().__init__(opt, shared)

class BahasaTeacher(FlowTeacher):
    pass


class DefaultTeacher(FlowTeacher):
    pass

    # def __init__(self, opt, shared=None):
    #     opt = copy.deepcopy(opt)

    #     # get datafile
    #     opt['parlaidialogteacher_datafile'] = _path(opt)

    #     super().__init__(opt, shared)