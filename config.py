import os
import os.path as osp
import argparse

from easydict import EasyDict as edict

_C = edict()

_C.buffer_length = 102
_C.min_hits = 1

_C.base_dir = "..."
_C.base_dir = "..."
_C.base_dir = "..."

def make_cfg():
    return _C