import pandas as pd
import numpy as np

import os, gc, glob
from collections import Counter
import itertools

DATA_DIR = os.getenv('DATA_DIR')
OUTPUT_DIR = os.getenv('OUTPUT_DIR')
CONFIG_DIR = os.getenv('CONFIG_DIR')


def _suggest_clicks(df, covisit_dict):
    type_weight_multipliers = {0: 1, 1: 6, 2: 3}

    # USE USER HISTORY AIDS AND TYPES
    aids=df.aid.tolist()
    types = df.type.tolist()
    unique_aids = list(dict.fromkeys(aids[::-1] ))
    # RERANK CANDIDATES USING WEIGHTS
    if len(unique_aids)>=20:
        weights=np.logspace(0.1,1,len(aids),base=2, endpoint=True)-1
        aids_temp = Counter() 
        # RERANK BASED ON REPEAT ITEMS AND TYPE OF ITEMS
        for aid,w,t in zip(aids,weights,types): 
            aids_temp[aid] += w * type_weight_multipliers[t]
        sorted_aids = [k for k,v in aids_temp.most_common(20)]
        return sorted_aids
    
    # USE "CLICKS" CO-VISITATION MATRIX
    aids2 = list(itertools.chain(*[covisit_dict[aid] for aid in unique_aids if aid in covisit_dict]))
    # RERANK CANDIDATES
    top_aids2 = [aid2 for aid2, cnt in Counter(aids2).most_common(20) if aid2 not in unique_aids]    
    result = unique_aids + top_aids2[:20 - len(unique_aids)]
    # USE TOP20 TEST CLICKS
    return result

def _past_aid_only(df):
    '''過去aidのみ
    '''
    type_weight_multipliers = {0: 1, 1: 6, 2: 3}

    # USE USER HISTORY AIDS AND TYPES
    aids=df.aid.tolist()
    types = df.type.tolist()
    unique_aids = list(dict.fromkeys(aids[::-1] ))
    if len(unique_aids)<20:
        return unique_aids

    else:
        # RERANK CANDIDATES USING WEIGHTS
        weights=np.logspace(0.1,1,len(aids),base=2, endpoint=True)-1
        aids_temp = Counter() 
        # RERANK BASED ON REPEAT ITEMS AND TYPE OF ITEMS
        for aid,w,t in zip(aids,weights,types): 
            aids_temp[aid] += w * type_weight_multipliers[t]
        sorted_aids = [k for k,v in aids_temp.most_common(20)]
        return sorted_aids



