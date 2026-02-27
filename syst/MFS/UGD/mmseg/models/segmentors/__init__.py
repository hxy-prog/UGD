# Copyright (c) OpenMMLab. All rights reserved.
from .base import BaseSegmentor
from .cascade_encoder_decoder import CascadeEncoderDecoder
from .encoder_decoder import EncoderDecoder
from .ddp import DDP
from .self_aligned_ddp import SelfAlignedDDP
from .dddp import DDDP
from .half_ddp import half_DDP
from .dddp2 import DDDP2
from .constrain_ddp import CDDP
# from .vaeddp import VAEDDP
from .ddp4n import DDP4n
from .encoder_decoder_c import CEncoderDecoder
from .constrin_tddp import CTDDP
from .encoder_decoder_ct import CtEncoderDecoder
from .no_ddp import nDDP
from .ncddp import nCDDP

__all__ = ['BaseSegmentor', 'EncoderDecoder', 'CascadeEncoderDecoder',
           'DDP', 'SelfAlignedDDP','DDDP','half_DDP','DDDP2','CDDP','DDP4n','CEncoderDecoder','CtEncoderDecoder','CTDDP','nDDP','nCDDP']
