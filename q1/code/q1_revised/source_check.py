"""Verify source-file hashes and clocks without fitting to the test split."""
import hashlib
import os
from pathlib import Path
import json
from pathlib import PureWindowsPath
from zipfile import ZipFile
import numpy as np
import pandas as pd
from audit_position import DATA,ROOT,OUT,save

def main():
    refs=pd.read_csv(DATA/"input_sha256.csv")
    refs["member"]=[PureWindowsPath(*PureWindowsPath(x).parts[1:]).as_posix() for x in refs.path]
    refs["source"]=refs.member.str.split("/").str[0]
    for source,entries in refs.groupby("source"):
        with ZipFile(next(Path(os.environ.get("Q1_ARCHIVES", ROOT/"C_Problem_Data")).rglob(source+".zip"))) as archive:
            for r in entries.itertuples():
                digest=hashlib.sha256()
                with archive.open(r.member) as f:
                    for b in iter(lambda:f.read(1024*1024),b""):digest.update(b)
                assert digest.hexdigest()==r.sha256 and archive.getinfo(r.member).file_size==r.bytes
    usage=pd.read_csv(OUT/"record_usage.csv");clocks=[]
    for r in pd.read_csv(DATA/"record_index.csv").itertuples():
        with np.load(DATA/"records"/f"{r.episode_uuid}.npz") as z:
            streams=[k[:-5] for k in z.files if k.endswith("__t_s")]
            ref=z["left_hand__t_s"]
            for ch in streams:
                t=z[ch+"__t_s"];delta=np.diff(t)
                exact=np.array_equal(t,ref)
                if r.dataset!="EgoDemo":
                    # Shared integer origin was persisted in the official record index.
                    origin=int(r.time_basis.split("origin ")[1].split(" ns")[0])
                    np.testing.assert_array_equal((z[ch+"__timestamp_ns"]-origin)/1e9,t)
                clocks.append(dict(record=r.episode_uuid,source=r.dataset,env=r.environment_id,split=r.split,
                                   channel=ch,aligned=exact,duplicate=int((delta==0).sum()),reversed=int((delta<0).sum())))
    f=pd.DataFrame(clocks);save(f,"clocks.csv")
    q=pd.read_csv(OUT/"quality_predictions.csv.gz")
    n=q.groupby(["episode_uuid","side"]).size().rename("quality_anchors").reset_index().rename(columns={"episode_uuid":"record"})
    usage=usage.merge(n,on=["record","side"],how="left");usage.quality_anchors=usage.quality_anchors.fillna(0).astype(int)
    inter=pd.read_csv(DATA/"natural_unusable_intervals.csv").rename(columns={"episode_uuid":"record","channel":"side"})
    inter=inter.merge(usage[["record","side","source","env"]],on=["record","side"],validate="many_to_one")
    inter["use"]=np.where(inter.split=="test","structure_only","natural_quality_or_recovery")
    save(inter,"interval_usage.csv")
    count=inter.groupby(["record","side"]).size().rename("unusable_intervals").reset_index()
    usage=usage.merge(count,on=["record","side"],how="left").fillna({"unusable_intervals":0})
    save(usage,"record_roles.csv")
    summary=dict(verified_source_hashes=len(refs),hand_timestamps=int(usage.n.sum()),eligible=int(usage.reference.sum()),
                 bad=int(usage.bad.sum()),unknown=int(usage.unknown.sum()),gaps=int(usage.gaps.sum()),records=int(usage.record.nunique()),
                 channels=len(f),aligned=int(f.aligned.sum()),duplicates=int(f.duplicate.sum()),reversed=int(f.reversed.sum()),
                 natural_intervals=inter.groupby("split").size().to_dict())
    (OUT/"source_check.json").write_text(json.dumps(summary,indent=2))
    print(summary)
if __name__=="__main__":main()
