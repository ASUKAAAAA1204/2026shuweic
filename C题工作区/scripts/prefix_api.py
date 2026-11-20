"""Trusted evaluator calls load_prefix; never pass unit_id/path to the policy."""
from pathlib import Path
import sys
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'运行依赖314'))
import numpy as np
ALLOWED_LENGTHS=(.2,.6,1.)
def load_prefix(unit_id,length):
 if length not in ALLOWED_LENGTHS:raise ValueError('length must be .2, .6 or 1.0')
 if not isinstance(unit_id,str) or not unit_id.startswith('U_') or not unit_id[2:].isalnum():raise ValueError('invalid private key')
 with np.load(R/'处理结果/v1/private'/(unit_id+'_'+str(float(length))+'.npz'),allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
