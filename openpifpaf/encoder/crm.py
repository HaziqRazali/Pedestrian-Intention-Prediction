import logging
import re

import numpy as np
import scipy.ndimage
import torch

from .annrescaler import AnnRescaler
from .encoder import Encoder
from ..utils import create_sink, mask_valid_area

# Not in use because the pre-processing for crm is in datasets.py
class Crm(Encoder):

    def __init__(self, head_name, stride, *, n_keypoints=None, **kwargs):
        return

    @staticmethod
    def match(head_name):
        return head_name in (
            'crm',
        )

    @classmethod
    def cli(cls, parser):
        return

    @classmethod
    def apply_args(cls, args):
        return

    def __call__(self, anns, width_height_original):
        return