"""Native extension for expected completed-weight regret."""
from functools import lru_cache
import ctypes as ct
import hashlib
import os
from pathlib import Path
import subprocess
import tempfile
import numpy as np
from bellman import library


@lru_cache(maxsize=1)
def extended():
    source=Path(__file__).resolve().with_name("guard.cpp")
    digest=hashlib.sha256(source.read_bytes()+(source.parent.parent/"dynamic.cpp").read_bytes()).hexdigest()
    path=Path(tempfile.gettempdir())/("q3_guard_"+digest[:16]+".dll")
    if not path.exists():
        tmp=path.with_name(path.stem+str(os.getpid())+".dll")
        subprocess.run(["g++","-std=c++17","-O3","-static","-shared",str(source),"-o",str(tmp)],check=True)
        os.replace(tmp,path)
    lib=ct.CDLL(str(path))
    for name in ("fit_value","best_job","free_value","refit_value"):
        f=getattr(lib,name);old=getattr(library(),name)
        f.argtypes=old.argtypes;f.restype=old.restype
    arr=np.ctypeslib.ndpointer(np.float64,flags="C_CONTIGUOUS")
    lib.guarded_best.argtypes=[ct.c_void_p]*2+[ct.c_int]*2+[ct.c_double]*2+[ct.c_int,arr]+[ct.c_double]*3
    lib.guarded_best.restype=ct.c_int
    return lib
