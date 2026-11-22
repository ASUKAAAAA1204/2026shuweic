"""Create reviewable acceptance copies without modifying research sources."""
from pathlib import Path

HERE = Path(__file__).resolve().parent
BREAK = HERE.parents[1]
DEST = HERE / "frozen"

MC = '''def mc_eval(plan, t0, zxy0, cost0, jobs, pdists, preds, H, B, samples, K, zk, Dmat, z0, done=()):
    """Sample-average legal fixed-plan replay, with no hidden-duration screening.

    Release occurs before travel/changeover. An affordable, released, chain-legal
    job is attempted if reserve fits, even if its sampled duration will overrun H.
    Such an attempt charges cost and exhausts the sample's remaining horizon.
    """
    t = np.full(K, float(t0))
    cost = np.full(K, float(cost0))
    val = np.zeros(K)
    z = np.full(K, z0, int)
    completed = {jid: np.full(K, jid in done, bool) for jid in jobs}
    for jid in plan:
        j = jobs[jid]
        k = zk[jid]
        D = Dmat[z, k]
        fee = 2 + .1 * .2 + .2 * D
        pred_ok = np.ones(K, bool)
        for p in preds[jid]:
            pred_ok &= completed[p]
        decision = np.maximum(t, j['release_s'])
        s = decision + D + .5
        end = s + samples[jid]
        attempted = pred_ok & (cost + fee <= B + 1e-12) & (s + .2 <= H + 1e-12)
        served = attempted & (end <= H + 1e-12)
        val += np.where(served & (end <= min(j['due_s'], H) + 1e-12), j['weight'], 0.0)
        t = np.where(attempted, np.minimum(H, s + np.maximum(samples[jid], .2)), t)
        # A skipped job cannot move the clock back or advance it to its release.
        z = np.where(attempted, k, z)
        cost = cost + np.where(attempted, fee, 0.0)
        completed[jid] = served
    return float(np.mean(val))


'''

FIRST = '''def _first_actionable(plan, jobs, o, B, H):
    # The action must be the very first item of the plan that was evaluated.
    # Selecting a later released job would silently execute a different plan.
    if not plan:
        return None
    jid = plan[0]
    j = jobs[jid]
    done = {k for k, v in jobs.items() if v['completed']}
    D = math.dist(np.asarray(o['zone_xy'], float), np.asarray(j['zone_xy'], float))
    if j['attempted'] or not set(j['predecessors']) <= done:
        return None
    if j['release_s'] > o['time'] + 1e-12:
        return None
    if o['cost'] + 2 + .02 + .2 * D > B + 1e-12:
        return None
    if o['time'] + D + .5 + .2 > H + 1e-12:
        return None
    return jid


'''


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    b = BREAK / "parallel" / "B_联合场景前瞻" / "code" / "joint_policy.py"
    (DEST / "b_policy.py").write_text(b.read_text(encoding="utf8"), encoding="utf8")
    e = BREAK / "parallel" / "E_稳健分布与选参" / "code" / "robust_chain_policy.py"
    text = e.read_text(encoding="utf8")
    text = text.replace(
        "  * chain value is gated per job by its *ancestor* completion probability, so\n    parallel branches are not multiplied together;",
        "  * root scenarios respect actual completion; the tail is a mean-time\n    heuristic with deterministic predecessor unlocking, not a chain-probability model;")
    text = text.replace('"""Mean-time greedy rollout; per-job value gated by ancestor completion probability."""',
                        '"""Mean-time greedy tail; predecessor unlocking is deterministic.\n\n    The legacy pcomp/ancestor loop is inert for available jobs. It is retained\n    only to preserve the frozen numerical behavior; do not attribute gains to it.\n    """')
    (DEST / "e_policy.py").write_text(text, encoding="utf8")
    c = BREAK / "parallel" / "C_尾部价值DP" / "code" / "dp_tail_policy.py"
    text = c.read_text(encoding="utf8")
    (DEST / "c_original.py").write_text(text, encoding="utf8")
    # The original surrogate DP values do not certify a stochastic lower bound.
    start = text.index("Reduction and its bound")
    end = text.index('\n"""', start)
    text = text[:start] + "Surrogate scope\n---------------\nThe time-bucketed fixed-order DP proposes plans. Its surrogate value is not a\ncertified lower or upper bound on the stochastic optimum. Corrected legal MC\nreplay ranks these plans with finite-sample uncertainty.\n" + text[end:]
    start = text.index("def mc_eval(")
    end = text.index("def _first_actionable(", start)
    text = text[:start] + MC + text[end:]
    start = text.index("def _first_actionable(")
    end = text.index("def repair_front(", start)
    text = text[:start] + FIRST + text[end:]
    (DEST / "c_policy.py").write_text(text, encoding="utf8")
    print("Frozen B, documented E, original C, and repaired C written", DEST)


if __name__ == "__main__":
    main()
