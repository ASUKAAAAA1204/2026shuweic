"""Priority policy for the published nominal cases, using public state only."""
from math import dist


def choose(obs,model):
    n=len(model["jobs"])
    p=model["params"]
    done={j["job_id"] for j in obs["jobs"] if j["completed"]}
    best=None
    best_score=float("-inf")
    future=[]
    by_id={j["job_id"]:j for j in obs["jobs"]}
    for i,key in enumerate(model["jobs"]):
        j=by_id[key]
        if j["attempted"]:continue
        if j["release_s"]>obs["time"]+1e-12:
            future.append(j["release_s"])
            continue
        if not set(j["predecessors"])<=done:continue
        d=dist(obs["zone_xy"],j["zone_xy"])
        if obs["time"]+d+.7>obs["H"]+1e-12 or obs["cost"]+2.02+.2*d>obs["B"]+1e-12:continue
        duration=d+.5+p[n+4]*model["typical"][j["family"]]
        slack=j["due_s"]-obs["time"]-duration
        late=max(0.,min(2.,-slack/8.))
        urgency=1/(1+max(0.,slack)/12.)
        score=p[i]+p[n]*urgency-p[n+1]*late-p[n+2]*d+p[n+3]*j["weight"]/(1+duration/5.)
        if score>best_score:
            best=key
            best_score=score
    if best is not None:return dict(type="execute",job_id=best,reserve=.2)
    return dict(type="wait",duration=min(future)-obs["time"]) if future else dict(type="stop")
