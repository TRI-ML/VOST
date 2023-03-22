import os
from .default import DefaultEngineConfig


class EngineConfig(DefaultEngineConfig):
    def __init__(self, exp_name='default', model='AOTT'):
        super().__init__(exp_name, model)
        self.STAGE_NAME = 'pre_vost'

        self.init_dir()

        self.DATASETS = ['vost']
        self.TRAIN_TOTAL_STEPS = 20000
        self.DATA_SEQ_LEN = 15
        self.TRAIN_LONG_TERM_MEM_GAP = 4
        self.MODEL_LINEAR_Q = False
        self.MODEL_JOINT_LONGATT = False
        self.MODEL_SIMPLIFIED_STM = True
        self.MODEL_RECURRENT_STM = True
        self.MODEL_IGNORE_TOKEN = True

        self.PRETRAIN_FULL = True  # if False, load encoder only
        self.PRETRAIN_MODEL = os.path.join('pretrain_models', 'pre_ytb.pth')