"""Construct and verify the five additive baseline-to-final bridges."""
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile
import sys

import fitz
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/"final_v1_evidence_extension"))
from plot_baseline_stability_lines import CASES, RESULTS, SPACE, VERSIONS, q3_results
from plot_five_dimensions import ROOT
from schedule_statistics import paired

OUT = ROOT/"figures/final_v1_waterfalls"


def weighted(frame, weights=None):
    if weights is None:
        weights = np.full(len(frame),1/len(frame))
    weights = np.asarray(weights)
    assert frame.notna().all().all() and np.isclose(weights.sum(),1.)
    start,end = frame[VERSIONS].to_numpy().T @ weights
    return frame.index.tolist(),(frame["终稿"]-frame["基线"]).to_numpy()*weights,start,end


def sensitivity(q):
    rows=[]
    for filename,index,easy,hard in [("q1_horizons.csv","horizon",.2,1),("q2_cost.csv","budget",2,0)]:
        curve=pd.read_csv(RESULTS/filename).query("split=='test'").pivot(index=index,columns="model",values="nade")
        rows.append((curve.loc[hard]-curve.loc[easy]).rename({"original":"基线","B":"终稿"}))
    rate=q.query("beta==.5").groupby("alpha")[VERSIONS].mean()
    rows.append(rate.loc[1]-rate.loc[1.5])
    frame=pd.DataFrame(rows,index=["一问预测时长\n0.2 → 1 秒","二问观测次数\n2 → 0 次","三问工期倍率\n1 → 1.5"])
    assert (frame>0).all().all()
    relative=frame.div(frame["基线"],axis=0)*100
    return ("参数敏感性","三问各以基线在指定扰动下的性能损失为 100，等权汇总。指数越低，表示整体损失越小。",
            *weighted(relative),"指数点",2,
            "前两问取固定阻尼与终稿的测试窗 NADE 增量，第三问取上一正式版与终稿的按时率降幅。每步为本问相对指数差的三分之一。")


def environment():
    keys=["split","environment_id","episode_uuid","side","window_id"]
    raw=pd.read_csv(RESULTS/"q2_windows.csv.gz").query("split in ['validation','test']")
    assert raw.requests.eq(2).all() and np.allclose(raw.waiting_s,.2)
    raw=raw.pivot(index=keys,columns="model",values="nade")[["original","B"]]
    assert raw.shape==(602,2) and raw.notna().all().all()
    for group in [keys[:-1],keys[:-2],keys[:2]]:
        raw=raw.groupby(level=group).mean()
    sites=pd.concat([raw.loc["validation"],raw.loc["test"]])
    names={"restaurant":"餐厅","supermarket":"超市","station":"快递站"}
    sites.index=[("开发" if i<4 else "测试")+f" · {s.split('_')[-1]}\n{names[s.split('_')[-2]]}" for i,s in enumerate(sites.index)]
    return ("跨环境稳定性","第二问在八个既有地点使用相同窗口和观测预算。整体误差按地点等权，NADE 越低越好。",
            *weighted(sites.rename(columns={"original":"基线","B":"终稿"})),"NADE",5,
            "开发集 212 窗、既有测试集 390 窗。每步为该地点的终稿与固定阻尼误差差值除以 8。")


def uncertainty(q):
    idx=q.set_index(["alpha","beta","case_id","seed"])[VERSIONS].sort_index()
    delta=idx.loc[1.5,.5]-idx.loc[1.,.5]
    old,new=[delta[m].unstack("case_id")[CASES].copy() for m in VERSIONS]
    assert old.shape==new.shape==(30,3) and old.index.equals(new.index)
    hybrid=old.copy()
    def interval(frame):
        return paired("long",frame.mean(axis=1)/100)
    first,last=interval(old),interval(new)
    widths=[first["halfwidth_pp"]]
    for case in CASES:
        hybrid[case]=new[case]
        widths.append(interval(hybrid)["halfwidth_pp"])
    np.testing.assert_allclose(widths[-1],last["halfwidth_pp"])
    caption=(f"基线 95% 区间 [{first['lower_pp']:.2f}, {first['upper_pp']:.2f}]，终稿 [{last['lower_pp']:.2f}, {last['upper_pp']:.2f}]（百分点）。"
             "中间步的大小取决于案例替换顺序。")
    return ("配对不确定性","指标为工期延长后按时率变化的配对 95% 区间半宽。依次换入三个案例，每步对相同 30 个工期种子重抽 10,000 次。",
            [f"换入 {c}" for c in CASES],np.diff(widths),widths[0],widths[-1],"百分点",3,caption)


def tail(q):
    nominal=q.query("alpha==1 and beta==.5")
    assert nominal.groupby("case_id").size().eq(30).all()
    frame=nominal.groupby("case_id")[VERSIONS].agg(lambda x:x.nsmallest(3).mean()).loc[CASES]
    return ("尾部风险","第三问名义工况：每个案例分别取最差 3 次，计算其平均加权按时率，再对三个案例等权。越高越好。",
            *weighted(frame),"百分点",2,
            "两版使用相同的 30 个官方工期种子。各案例分别选取本版本的最差一成，各步为案例尾均值差的三分之一。")


def seeds(q):
    results=HERE.parent
    nominal=pd.read_csv(HERE/"previous_nominal_trials.csv.gz")
    long=pd.read_csv(results/"q3_baseline_seed_control/trials.csv.gz").query("alpha==1.5 and beta==.5")
    old=pd.concat([nominal,long],ignore_index=True)
    final=pd.concat([pd.read_csv(f).assign(batch=b) for b,f in [
        ("original",results/"final_v1_stability/seed_trials.csv.gz"),
        ("replication",results/"final_v1_evidence_extension/internal_replication.csv.gz")]]).query("alpha in [1.,1.5] and beta==.5")
    keys=["batch","alpha","beta","case_id","seed","internal_seed"]
    frames=[g.set_index(keys).sort_index() for g in (old,final)]
    assert len(old)==len(final)==7200 and frames[0].index.is_unique and frames[0].index.equals(frames[1].index)
    assert frames[0].instance_hash.equals(frames[1].instance_hash)
    defaultkey=["alpha","beta","case_id","seed"]
    official=q.query("alpha in [1.,1.5] and beta==.5").set_index(defaultkey).sort_index()
    for ver,g in zip(VERSIONS,[old,final]):
        scores=g.query("internal_seed==20261122").set_index(defaultkey).sort_index()
        assert official.index.equals(scores.index) and len(scores)==180 and g.valid_protocol.all() and g.invalid_actions.eq(0).all()
        np.testing.assert_allclose(official[ver],scores.weighted_on_time_rate*100,atol=1e-9)
    stats={}
    for ver,g in zip(VERSIONS,frames):
        samples=g.weighted_on_time_rate.groupby(level=keys[:-1])
        assert samples.size().eq(20).all()
        # Center before std so that equal rates return exactly zero.
        s=samples.agg(lambda x:(x-x.iloc[0]).std(ddof=1)*100)
        stats[ver]=s.groupby(level=["alpha","batch"]).mean()
    frame=pd.DataFrame(stats).loc[[(a,b) for a in [1.,1.5] for b in ["original","replication"]]]
    frame.to_csv(HERE/"matched_seed_sd.csv")
    frame.index=["名义\n首批","名义\n复验批","长工期\n首批","长工期\n复验批"]
    return ("内部抽样种子稳定性","以单个固定实例跨 20 个内部种子的按时率标准差衡量波动。名义、长工期及两批次等权，标准差越小越稳定。",
            *weighted(frame),"百分点",3,
            "每档 90 个固定实例。上一正式版名义工况补测 3,600 次，长工期复用 3,600 次配对回放。终稿固定 K = 64。")


def main():
    from draw import waterfall
    OUT.mkdir(parents=True,exist_ok=True)
    q=q3_results()
    records=[]
    with fitz.open() as book, ZipFile(OUT/"五张高清PNG.zip","w",ZIP_DEFLATED) as archive:
        for n,plate in enumerate([sensitivity(q),environment(),uncertainty(q),tail(q),seeds(q)],1):
            title,subtitle,labels,changes,start,end,unit,digits,foot=plate
            np.testing.assert_allclose(start+sum(changes),end,atol=1e-10)
            for label,change in zip(labels,changes):
                records.append(dict(figure=n,metric=title,unit=unit,baseline=start,final=end,step=label.replace("\n"," "),change=change))
            endpoint_unit="%" if title=="尾部风险" else unit
            fig=waterfall(title,subtitle,labels,changes,start,end,endpoint_unit,digits,foot)
            if title=="尾部风险":
                fig.axes[0].set_ylabel("相对基线的变化（百分点）")
            assert len(fig.axes)==1
            stem=f"{n:02d}_{title}瀑布图"
            pdf,png=OUT/(stem+".pdf"),OUT/(stem+".png")
            fig.savefig(pdf,metadata={"Title":title})
            fig.savefig(png,dpi=450)
            import matplotlib.pyplot as plt
            plt.close(fig)
            with fitz.open(pdf) as page:
                book.insert_pdf(page)
            archive.write(png,arcname=png.name)
        book.save(OUT/"五维瀑布图审核稿.pdf",garbage=4,deflate=True)
    pd.DataFrame(records).to_csv(HERE/"waterfall_values.csv",index=False)


if __name__=="__main__":
    main()
