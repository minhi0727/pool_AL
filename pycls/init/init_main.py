import numpy as np 
import torch
import os
import sys

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

add_path(os.path.abspath('/home/chominhi/work/init-pools-dal-main'))



from sampling import SelfSupervisionSampling, ClusteringSampling    #원래 sampilng안에 점 찍혀었음; 
import pycls.utils.logging as lu
import os

logger = lu.get_logger(__name__)

class InitialPool:
    """
    Implements initial pool sampling methods.
    """

    def __init__(self, cfg):
        self.cfg = cfg

    def sample_from_uSet(self, dataset):
        """
        Sample from uSet using cfg.INIT_POOL.SAMPLING_FN.

        INPUT
        ------
        dataset: PyTorch dataset object

        OUTPUT
        -------
        Returns initSet, uSet
        """
        assert (self.cfg.INIT_POOL.INIT_RATIO > 0) & (self.cfg.INIT_POOL.INIT_RATIO < 1) , "Expected a label ration between 0 and 1"

        budgetSize = int((self.cfg.INIT_POOL.INIT_RATIO)*len(dataset))

        if self.cfg.INIT_POOL.SAMPLING_FN in ['simclr', 'vae']:
            initSet, uSet = SelfSupervisionSampling(dataset=dataset, budgetSize=budgetSize, 
                sampling_fn=self.cfg.INIT_POOL.SAMPLING_FN, dataset_name=self.cfg.DATASET.NAME).sample()
        elif self.cfg.INIT_POOL.SAMPLING_FN in ['scan', 'kmeans']:
            initSet, uSet = ClusteringSampling(dataset=dataset, budgetSize=budgetSize,  
                sampling_fn=self.cfg.INIT_POOL.SAMPLING_FN, dataset_name=self.cfg.DATASET.NAME).sample()

        return initSet, uSet