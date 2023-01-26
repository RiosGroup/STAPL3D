import os
from stapl3d import Stapl3r

class Block3r(Stapl3r):
    _doc_main = """"""
    _doc_attr = """"""
    _doc_exam = """"""
    __doc__ = f"{_doc_main}{Stapl3r.__doc__}{_doc_attr}{_doc_exam}"

    def __init__(self, image_in='', parameter_file='', **kwargs):

        if 'module_id' not in kwargs.keys():
            kwargs['module_id'] = 'blocks'

        super(Block3r, self).__init__(
            image_in, parameter_file,
            **kwargs,
            )

        self._fun_selector = {
            'blockinfo': self.write_blockinfo,
            }

        self._parallelization = {
            'blockinfo': ['blocks'],
            }

        self._parameter_sets = {
            'blockinfo': {
                'fpar': self._FPAR_NAMES,
                'ppar': ('fullsize', 'blocksize', 'blockmargin',
                         'boundary_truncation',
                         'shift_final_block_inward', 'pad_kwargs'),
                'spar': ('_n_workers', 'blocks'),
                },
            }

        self._parameter_table = {
        }

        default_attr = {
        }
        for k, v in default_attr.items():
            setattr(self, k, v)

        for step in self._fun_selector.keys():
            step_id = 'blocks' if step=='blockinfo' else self.step_id
            self.set_parameters(step)

        self._init_paths()

        self._init_log()

        self._prep_blocks()

        self.view_block_layout = []  # ['fullsize', 'margins', 'blocks']
        self._images = []
        self._labels = []

    def _init_paths(self):
        """Set input and output filepaths."""

        if '{f' in self.image_in:
            prefixes, suffix = [''], 'f'
        else:
            prefixes, suffix = [self.prefix, 'blocks'], 'b'

        bpat = self._build_path(
            moduledir='blocks',
            prefixes=prefixes,
            suffixes=[{suffix: 'p'}],
            rel=False,
            )

        blockdir = os.path.join(self.datadir, 'blocks')
        os.makedirs(blockdir, exist_ok=True)

        self._paths = {
            'blockinfo': {
                'inputs': {
                    'data': self.image_in,
                    },
                'outputs': {
                    'blockfiles': f"{bpat}.h5",
                },
            },
        }

        for step in self._fun_selector.keys():
            step_id = 'blocks' if step=='blockinfo' else self.step_id
            self.inputpaths[step]  = self._merge_paths(self._paths[step], step, 'inputs', step_id)
            self.outputpaths[step] = self._merge_paths(self._paths[step], step, 'outputs', step_id)
