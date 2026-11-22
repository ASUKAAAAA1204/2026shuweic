import math

def independent_rebuild(jobs, events, zones, horizon, budget, observation_limit):
    """Recompute legality, travel, elapsed time, costs, chains and reward."""
    by_id = {j["job_id"]: j for j in jobs}
    attempted, completed = set(), {}
    clock = cost = 0.0
    zone = tuple(zones["Z0"])
    reviews = invalid = censored = 0
    overrun = idle = 0.0
    violations = []
    for index, event in enumerate(events):
        action = event["action"]
        typ = action.get("type")
        t0 = clock
        if abs(event["decision_time"] - t0) > 1e-9:
            violations.append((index, "decision_time_mismatch"))
        if typ == "stop":
            continue
        if typ == "wait":
            duration = action.get("duration")
            if not isinstance(duration, (int, float)) or duration <= 0 or t0 + duration > horizon + 1e-9:
                violations.append((index, "illegal_wait")); invalid += 1
                continue
            clock = min(horizon, t0 + duration)
            continue
        jid = action.get("job_id")
        j = by_id.get(jid)
        if j is None or jid in attempted or j["release_s"] > t0 + 1e-9:
            violations.append((index, "unavailable_or_repeated_job")); invalid += 1
            continue
        if typ == "review":
            reviews += 1
            if reviews > observation_limit or t0 + 1 > horizon + 1e-9 or cost + 1 > budget + 1e-9:
                violations.append((index, "review_limit")); invalid += 1
                continue
            cost += 1.
            clock = t0 + 1.
        elif typ == "execute":
            if any(pred not in completed for pred in j["predecessors"]):
                violations.append((index, "predecessor_not_completed")); invalid += 1
                continue
            reserve = action.get("reserve")
            if not isinstance(reserve, (int, float)) or reserve < .2:
                violations.append((index, "invalid_reserve")); invalid += 1
                continue
            dest = tuple(zones[j["zone_id"]])
            distance = math.dist(zone, dest)
            start = t0 + distance + .5
            fee = 2. + .1 * reserve + .2 * distance
            if start + reserve > horizon + 1e-9 or cost + fee > budget + 1e-9:
                violations.append((index, "infeasible_execution")); invalid += 1
                continue
            if abs(start - event["service_start"]) > 1e-8 or abs(distance - event["distance"]) > 1e-8:
                violations.append((index, "physics_mismatch"))
            if abs(fee - event["fee"]) > 1e-8:
                violations.append((index, "fee_mismatch"))
            realized = event["realized_duration"]
            finish = start + realized
            attempted.add(jid)
            if finish <= horizon + 1e-9:
                completed[jid] = finish
                overrun += max(realized - reserve, 0.)
                idle += max(reserve - realized, 0.)
            else:
                censored += 1
                overrun += max(min(realized, horizon - start) - reserve, 0.)
            clock = min(horizon, start + max(realized, reserve))
            cost += fee
            zone = dest
        else:
            violations.append((index, "unknown_action")); invalid += 1
            continue
        if abs(clock - event["after"]["time"]) > 1e-8:
            violations.append((index, "clock_mismatch"))

    total_weight = sum(j["priority_weight"] for j in jobs)
    ontime = sum(j["priority_weight"] for j in jobs
                 if j["job_id"] in completed and completed[j["job_id"]] <= j["due_s"] + 1e-9)
    done_weight = sum(j["priority_weight"] for j in jobs if j["job_id"] in completed)
    tardy = sum(j["priority_weight"] * max(completed.get(j["job_id"], horizon) - j["due_s"], 0.)
                for j in jobs)
    return {
        "weighted_on_time_rate": ontime / total_weight,
        "weighted_completion_rate": done_weight / total_weight,
        "weighted_tardiness_s": tardy,
        "cost": cost,
        "observations": reviews,
        "elapsed_s": clock,
        "observed_overrun_s": overrun,
        "reserved_idle_s": idle,
        "censored_units": censored,
        "invalid_actions": invalid,
        "violations": violations,
    }
