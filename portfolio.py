from typing import List
import pandas as pd

from subportfolio import Subportfolio
from assetuniverse import AssetUniverse

class Portfolio:
    def __init__(self, subportfolios: List[Subportfolio], leverage:float=1.0):
        self.target_weights = leverage*self.combine_weights(subportfolios)
        self.cash_weight = (1 - self.target_weights.sum(axis=1)).clip(0.0, 1.0)
    
    def combine_weights(self, subportfolios: List[Subportfolio]):
        target_weights = subportfolios[0].target_weights
        for i in range(1, len(subportfolios)):
            target_weights = target_weights.add(
                subportfolios[i].target_weights, 
                fill_value=0
                )
        return target_weights/len(subportfolios)
