"""Frozen Q1 position and Q2 paid-observation replay on the official split."""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

SPACE = Path(__file__).resolve().parents[2]
PACKAGE = SPACE / "独立方案B_旋转模型"
DELIVERY = PACKAGE / "第二问/delivery"
BASELINE = SPACE / "第二问最终方案"
PROTOCOL = SPACE / "阶段交付/stage1_review/corrected_results"
sys.path[:0] = [str(DELIVERY/"code"), str(BASELINE/"code")]
from rotation_inference import predict_model
from rotation_state import factory
from replay_engine import replay, MotionState, FixedPolicy
from run_complete import aggregate, radius
from continuous_range_state import fixed_joint_replay


def load_json(path):
    return json.loads(path.read_bytes())


def as_history(raw, anchor):
    past = {}
    for key, values in raw.items():
        clock = raw.get(key.split("__")[0]+"__t_s")
        past[key] = values[clock<=anchor] if clock is not None and values.ndim and len(values)==len(clock) else values
    return past


def forecast_position(parameters,raw,side,anchor_index,ages,source):
    """Predict absolute hand centroids from channels observed by the anchor."""
    anchor = raw[side+"__t_s"][anchor_index]
    origin = raw[side+"__position"][anchor_index].mean(axis=0)
    return origin+predict_model(parameters,as_history(raw,anchor),side,anchor,np.asarray(ages),source)


def question_one(window, ages, truth, anchor_position, initial_velocity, trajectory, scale):
    old = anchor_position + (-.2*np.expm1(-ages/.2))[:,None]*initial_velocity
    rows = []
    for model,pred in [("original",old),("B",trajectory)]:
        error = np.linalg.norm(pred-truth,axis=1)/scale
        for h in (.2,.6,1.):
            e = error[ages<=h+1e-12]
            assert len(e) and np.isfinite(e).all()
            rows.append(dict(window,model=model,horizon=h,nade=e.mean(),nfde=e[-1]))
    return rows, np.linalg.norm(old-truth,axis=1), np.linalg.norm(trajectory-truth,axis=1)


def replay_adopted(t,x,ht,hx,trajectory,scale,src,new_cfg,prior):
    """Paid simulation. The scheduler receives the frozen forecast and arrived payloads only."""
    radii = []
    class WithRange(factory(t,trajectory,MotionState)):
        def predict(self,now):
            pred = super().predict(now)
            params = new_cfg["return_models"][src] if self.returns else prior[src]
            radii.append(radius(params,now-self.last_time,np.linalg.norm(self.velocity),scale))
            return pred
    new = replay(t,x,ht,hx,FixedPolicy(),state_factory=WithRange)
    return new,radii


def question_two(meta, t, x, ht, hx, trajectory, scale, old_cfg, new_cfg, prior):
    src = meta["dataset"]
    old = fixed_joint_replay(t,x,ht,hx,source=src,scale=scale,
                             prior_models=old_cfg["prior_models"],return_models=old_cfg["return_models"])
    new,radii = replay_adopted(t,x,ht,hx,trajectory,scale,src,new_cfg,prior)
    assert old["actions"] == new["actions"]
    requests = [a for a in new["actions"] if a["event"]=="request"]
    returns = [a for a in new["actions"] if a["event"]=="return"]
    assert len(requests)==len(returns)==2
    for scheduled, sent, received in zip((.3,.6),requests,returns,strict=True):
        idx = np.flatnonzero(t-t[0]+1e-12>=scheduled)[0]
        assert sent["time"]==t[idx] and received["valid"]
        assert abs(received["time"]-sent["time"]-.1)<1e-12
        assert received["processed_at"]==t[np.flatnonzero(t+1e-12>=received["time"])[0]]
    rows, phases, horizons, budgets = [], [], [], []
    returned = np.array([p["returns"] for p in old["points"]])>0
    p_old = np.array([p["prediction"] for p in old["points"]])
    p_new = np.array([p["prediction"] for p in new["points"]])
    assert np.array_equal(returned,np.array([p["returns"] for p in new["points"]])>0)
    np.testing.assert_allclose(p_old[returned],p_new[returned],atol=1e-12,rtol=0)
    old_radius = np.array([q["radius"] for q in old["ranges"]])
    np.testing.assert_allclose(old_radius[returned],np.asarray(radii)[returned],atol=1e-12,rtol=0)
    for name,run,pred,rad in [("original",old,p_old,old_radius),("B",new,p_new,radii)]:
        e = np.linalg.norm(pred-x,axis=1)/scale
        r = np.asarray(rad)/scale
        assert np.isfinite(e).all() and np.isfinite(r).all()
        np.testing.assert_allclose(e,[p["error"]/scale for p in run["points"]],atol=1e-12,rtol=0)
        assert len(t)==len(r)
        rows.append(dict(meta,model=name,nade=e.mean(),nfde=e[-1],coverage=(e<=r).mean(),
                         width=2*r.mean(),pinball=np.maximum(.9*(e-r),-.1*(e-r)).mean(),
                         tolerance=(e<=.05).mean(),whole_coverage=float((e<=r).all()),
                         requests=run["metrics"]["requests"],waiting_s=run["metrics"]["waiting_s"]))
        for after in (False,True):
            use = returned==after
            phases.append(dict(meta,model=name,phase="after" if after else "before",
                               nade=e[use].mean(),coverage=(e[use]<=r[use]).mean()))
        for horizon in (0.,.05,.1,.2,.3,.4,.6,.8,1.):
            prefix=e[t-t[0]<=horizon+1e-9]
            horizons.append(dict(meta,model=name,horizon=horizon,nade=prefix.mean(),
                                 point_success=(prefix<=.05).mean(),whole_success=float((prefix<=.05).all())))
        for budget in (0,1,2):
            variant=run
            if budget<2:
                state=factory(t,trajectory,MotionState) if name=="B" else MotionState
                variant=replay(t,x,ht,hx,FixedPolicy(),budget=budget,state_factory=state)
            m=variant["metrics"]
            budgets.append(dict(meta,model=name,budget=budget,nade=m["ade"]/scale,nfde=m["fde"]/scale,
                                requests=m["requests"],waiting_s=m["waiting_s"]))
    return rows,phases,horizons,budgets


def fold_configs():
    old_model = BASELINE/"model"
    model = DELIVERY/"model"
    original = load_json(old_model/"model.json")
    current = load_json(model/"model.json")
    result = {"full": (original,current,load_json(model/"prior_models.json"),model/"rotation_full.npz")}
    def folds(path):
        return {(r["held"],r["dataset"]):r for r in load_json(path)}
    old_pr,old_ret = folds(old_model/"oof_prior.json"),folds(old_model/"oof_return.json")
    new_pr,new_ret = folds(model/"oof_prior_models.json"),folds(model/"oof_return.json")
    for held,scales in load_json(model/"fold_scales.json").items():
        def by_source(table):
            return {s:table[held,s] for s in scales if (held,s) in table}
        old = dict(scale=scales,prior_models=by_source(old_pr),return_models=by_source(old_ret))
        new = dict(scale=scales,return_models=by_source(new_ret))
        result[held] = (old,new,by_source(new_pr),model/f"oof_rotation/{held}_parameters.npz")
    return result


def run(processed, out):
    from range_calibration import calibrate
    from quality_recovery import cohort_outputs, finish_quality
    manifest = pd.read_csv(PROTOCOL/"data/windows.csv")
    records = pd.read_csv(processed/"record_index.csv")
    assert manifest.split.value_counts().to_dict()==dict(train=1527,test=390,validation=212)
    assert manifest.window_id.is_unique and records.episode_uuid.is_unique
    assert records.groupby("environment_id").split.nunique().max()==1
    meta_fields = ["episode_uuid","dataset","environment_id","split"]
    assert len(manifest.merge(records[meta_fields],on=meta_fields,validate="many_to_one"))==len(manifest)
    intervals = pd.read_csv(processed/"natural_unusable_intervals.csv")
    segments = pd.read_csv(processed/"segments_private.csv")
    # Both questions share the very same fully fitted rotation predictor.
    assert (PACKAGE/"第一问/model/model_parameters.npz").read_bytes()==(DELIVERY/"model/rotation_full.npz").read_bytes()
    assert (PACKAGE/"第一问/code/rotation_inference.py").read_bytes()==(DELIVERY/"code/rotation_inference.py").read_bytes()
    configs = fold_configs()
    weights, q1, q2, phases, points, qualities, recoveries = {}, [], [], [], [], [], []
    horizons,budgets,context=[],[],[]
    for uid, ws in manifest.groupby("episode_uuid"):
        with np.load(processed/"records"/f"{uid}.npz",allow_pickle=False) as f:
            raw = dict(f)
        record = ws.iloc[0][meta_fields].to_dict()
        spans=segments[segments.episode_uuid==uid]
        held = record["environment_id"] if record["split"]=="train" else "full"
        old,new,prior,params = configs[held]
        if held not in weights:
            with np.load(params,allow_pickle=False) as f:
                weights[held] = dict(f)
        scale = new["scale"][record["dataset"]]
        labels,events = cohort_outputs(record,raw,intervals[intervals.episode_uuid==uid],scale)
        qualities.extend(labels);recoveries.extend(events)
        for row in ws.itertuples():
            meta = dict(record,window_id=row.window_id,side=row.side)
            i,j = row.start_index,row.end_index_exclusive
            t = raw[row.side+"__t_s"]
            x = raw[row.side+"__position"].mean(1)
            assert raw[row.side+"__reference_candidate"][row.history_start:j].all()
            ages = t[i:j]-t[i-1]
            pos = forecast_position(weights[held],raw,row.side,i-1,ages,row.dataset)
            vel = (x[i-1]-x[i-2])/(t[i-1]-t[i-2])
            scores,err_old,err_new = question_one(meta,ages,x[i:j],x[i-1],vel,pos,scale)
            q1.extend(scores)
            points.append(pd.DataFrame(dict(**meta,age=ages,old_error=err_old,new_error=err_new,scale=scale)))
            scores,phase,curve,frontier = question_two(meta,t[i:j],x[i:j],t[i-2:i],x[i-2:i],pos,scale,old,new,prior)
            q2.extend(scores);phases.extend(phase)
            horizons.extend(curve);budgets.extend(frontier)
            hits=spans[(spans.start_s<=t[i])&(spans.end_s>t[i])]
            stage="unannotated" if hits.empty else "ambiguous"
            if len(hits)==1:
                s=hits.iloc[0]
                stage=("early","middle","late")[min(2,int(3*(t[i]-s.start_s)/s.duration_s))]
            context.append(dict(window_id=row.window_id,stage=stage,initial_speed=np.linalg.norm(vel)/scale))
    q1,q2 = pd.DataFrame(q1),pd.DataFrame(q2)
    q1.to_csv(out/"q1_windows.csv.gz",index=False)
    q2.to_csv(out/"q2_windows.csv.gz",index=False)
    group=["split","model","horizon","dataset","environment_id","episode_uuid"]
    summary=q1
    for n in (6,5,4,3,2):
        summary=summary.groupby(group[:n],as_index=False)[["nade","nfde"]].mean()
        if n==3:summary.to_csv(out/"q1_horizons.csv",index=False)
    summary.to_csv(out/"q1_summary.csv",index=False)
    metrics=["nade","nfde","coverage","width","pinball","tolerance","whole_coverage","requests","waiting_s"]
    aggregate(q2,["split","model"],metrics).to_csv(out/"q2_summary.csv",index=False)
    aggregate(pd.DataFrame(phases),["split","model","phase"],["nade","coverage"]).to_csv(out/"q2_phases.csv",index=False)
    aggregate(pd.DataFrame(horizons),["split","model","horizon"],["nade","point_success","whole_success"]).to_csv(out/"q2_prefix_5pct.csv",index=False)
    aggregate(pd.DataFrame(budgets),["split","model","budget"],["nade","nfde","requests","waiting_s"]).to_csv(out/"q2_cost.csv",index=False)
    ctx=pd.DataFrame(context).merge(manifest[["window_id","split","dataset"]],on="window_id",validate="one_to_one")
    cut=ctx.query("split=='train'").groupby("dataset").initial_speed.median()
    ctx["speed_group"]=np.where(ctx.initial_speed>ctx.dataset.map(cut),"above_train_median","at_or_below_train_median")
    annotated=q2.merge(ctx[["window_id","stage","speed_group"]],on="window_id",validate="many_to_one")
    for key in ["stage","speed_group"]:
        groups=["split","model",key]
        table=aggregate(annotated,groups,["nade","coverage"])
        table.merge(annotated.groupby(groups).size().rename("windows"),on=groups).to_csv(out/f"q2_{key}.csv",index=False)
    details=pd.concat(points,ignore_index=True)
    assert len(details)==65613 and not details.duplicated(["window_id","age"]).any()
    calibrate(details,out)
    finish_quality(qualities,recoveries,out,PROTOCOL)
    # Original Q2 must reproduce all archived windows, independently of their split.
    archived=pd.read_csv(BASELINE/"results/window_metrics.csv").set_index("window_id").sort_index()
    actual=q2.query("model=='original'").set_index("window_id").sort_index()
    assert actual.index.equals(archived.index)
    cols=["nade","nfde","coverage","width","pinball","requests","waiting_s"]
    difference=float((archived[cols]-actual[cols]).abs().max().max())
    assert difference<1e-10
    (out/"replay_audit.json").write_text(json.dumps(dict(windows=len(manifest),points=len(details),
        request_return_pairs=2*len(manifest),original_q2_max_window_difference=difference,
        training_folds=len(weights)-1),indent=2),encoding="utf8")
