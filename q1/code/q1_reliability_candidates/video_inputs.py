"""Build timestamp-aligned prefix covariates from the left head camera."""
import argparse
import hashlib
import io
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import NamedTemporaryFile, TemporaryDirectory
from zipfile import ZipFile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT/'C_Problem_Data/v1/v1'
OUT = ROOT/'results/q1_reliability_candidates'
DECODER = Path(__file__).parent/'video/decode.py'
FEATURES = ['gray','contrast','gradient','frame_delta']
VISUAL = FEATURES+[f'avg_{f}' for f in FEATURES]
KEY = ['episode_uuid','side','index']


def extract(movie, member, cache, python):
    if not cache.exists():
        with TemporaryDirectory() as temp:
            path = Path(temp)/'camera.mp4'
            with movie.open(member) as src, path.open('wb') as dst:
                shutil.copyfileobj(src,dst)
            dest = Path(temp)/'signals.npy'
            subprocess.run([str(python),'-B',str(DECODER.resolve()),str(path),str(dest)],check=True)
            with NamedTemporaryFile(dir=cache.parent,suffix='.part',delete=False) as tmp:
                np.save(tmp,np.load(dest,allow_pickle=False))
            Path(tmp.name).replace(cache)
    signals = np.load(cache,allow_pickle=False)
    assert signals.ndim==2 and signals.shape[1]==4 and np.isfinite(signals).all()
    return signals


def main(python):
    OUT.mkdir(exist_ok=True,parents=True)
    cache = OUT/'camera_cache'
    cache.mkdir(exist_ok=True)
    manifest = pd.read_csv(DATA/'record_index.csv').query("split in ['train','validation']")
    anchors = pd.read_csv(ROOT/'results/q1_reliability/predictions.csv.gz',usecols=KEY+['split'])
    assert manifest.episode_uuid.nunique()==62 and set(manifest.episode_uuid)==set(anchors.episode_uuid)
    digest = hashlib.sha256(DECODER.read_bytes()).hexdigest()[:12]
    records,coverage = [],[]
    for source,group in manifest.groupby('dataset'):
        archive = next((ROOT/'C_Problem_Data').rglob(source+'.zip'))
        with ZipFile(archive) as z:
            for row in group.itertuples():
                folder = f'{source}/{row.folder}/videos/'
                demo = source=='EgoDemo'
                member = folder+('observation.images.head_left/chunk-000/file-000.mp4' if demo else 'head_left.mp4')
                cached = cache/f'{row.episode_uuid}_{z.getinfo(member).CRC:x}_{digest}.npy'
                signals = extract(z,member,cached,python)
                if not demo:
                    frames = pd.read_csv(io.BytesIO(z.read(folder+'head_left_frames.csv')))
                    clock = frames['_log_time_ns'].to_numpy(np.int64)
                    np.testing.assert_array_equal(frames['_message_index'],np.arange(len(signals)))
                    assert len(clock)==len(signals) and (np.diff(clock)>0).all()
                block = anchors[anchors.episode_uuid==row.episode_uuid]
                with np.load(DATA/'records'/f'{row.episode_uuid}.npz',allow_pickle=False) as raw:
                    for side,sub in block.groupby('side'):
                        t = raw[side+'__t_s']
                        if demo:
                            ix = raw[side+'__frame_index']
                            np.testing.assert_array_equal(ix,np.arange(len(signals)))
                        else:
                            ts = raw[side+'__timestamp_ns']
                        for i in sub['index']:
                            if demo:
                                lo,end = np.searchsorted(t,t[i]-.2),ix[i]+1
                                lag = 0.
                            else:
                                lo,end = np.searchsorted(clock,ts[i]-200_000_000),np.searchsorted(clock,ts[i],side='right')
                                lag = (ts[i]-clock[end-1])*1e-9 if end>lo else np.nan
                            n = end-lo
                            vals = np.full(8,np.nan)
                            if n>=2:
                                a = signals[lo:end]
                                vals = np.r_[a[-1],a[:,:3].mean(axis=0),a[1:,3].mean()]
                                assert 0<=lag<=.2
                            records.append((row.episode_uuid,side,i,*vals,n,lag))
                coverage.append((row.episode_uuid,source,row.split,len(signals),len(block)))
            print('Decoded',source,len(group),flush=True)
    frame = pd.DataFrame(records,columns=KEY+VISUAL+['camera_frames','lag_s'])
    assert len(frame)==len(anchors) and not frame.duplicated(KEY).any()
    frame.to_csv(OUT/'video_inputs.csv.gz',index=False)
    pd.DataFrame(coverage,columns=['record','source','split','decoded','anchors']).to_csv(OUT/'camera_records.csv',index=False)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--decoder-python',default=sys.executable)
    main(parser.parse_args().decoder_python)
