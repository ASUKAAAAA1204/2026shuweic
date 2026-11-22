"""Consolidate every audit run into results/trials.csv + results/summary.json.

Only rows produced by code/audit_runner.py are used.  Holdout, oracle and the
chain-gating probe are computed by holdout.py / verify_oracle.py / probe_chain.py.
"""
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROLE = HERE.parent
RES = ROLE / "results"

TAGS = {
    "_validate": ["baseline", "audit_ref"],
    "A_full": ["baseline", "A_v2", "A_d2b4"],
    "A_d3_nominal": ["A_d3b4"],
    "B_full3": ["baseline", "B_K32", "B_K64", "B_K128"],
    "C_mc_full": ["baseline", "C_mc_t4_s64_dt10", "C_mc_t4_s64_dt05"],
    "C_s256": ["C_mc_t4_s256_dt05", "C_mc_t4_s256_dt10"],
    "BC_full": ["C_plan"],
    "DEF_full": ["baseline", "E_pooled_s20_fb3", "E_env_s40_fb3_w95", "E_env_s20_fb3",
                 "E_pooled_s20_fb3_w95", "D_v2_base", "D_v2_free", "D_v2_paid", "D_v2_both"],
    "E_selected": ["E_env_s40_fb3"],
    "official": ["official_s40"],
}
MAPPINGS = [(1., .5), (.75, .5), (1.5, .5), (1., 0.), (1., 1.)]
NOM = (1., .5)

SELECTION = {
    "baseline": "reference, no selection",
    "audit_ref": "audit-corrected reference (clock+chain+budget), strength=40 fixed by the audit; control only",
    "A_v2": "A's declared primary; depth=1/strength=20 fixed before validation",
    "A_d2b4": "A ablation; depth=2/beam=4",
    "A_d3b4": "A ablation; depth=3/beam=4, nominal mapping only",
    "B_K32": "B smoke-time primary K=32",
    "B_K64": "B main config: stated rule 'max K in required grid {8,16,32,64}'; chosen after seeing validation",
    "B_K128": "B supplementary; larger than required grid",
    "C_plan": "C default plan (dt=.5), no per-config tuning",
    "C_mc_t4_s64_dt05": "C earlier candidate (64 samples)",
    "C_mc_t4_s64_dt10": "C earlier candidate (64 samples, dt=1.0)",
    "C_mc_t4_s256_dt05": "C candidate (256 samples, dt=.5), not submitted",
    "C_mc_t4_s256_dt10": "C submitted config: max of its candidate table on the same validation seeds (post-hoc)",
    "official_s40": "official pre-audit lookahead at strength=40; control for the audit A/B (contains P1/P2 defects)",
    "D_v2_base": "D control: v2 with no information channel",
    "D_v2_free": "D free-prefix arm (gate resolved on train only)",
    "D_v2_paid": "D paid-review arm (gate resolved on train only)",
    "D_v2_both": "D free+paid arm",
    "E_env_s40_fb3": "E selected by train-environment grouped CV before validation; confirmatory",
    "E_env_s40_fb3_w95": "E candidate (CV tie-break rule), not selected",
    "E_env_s20_fb3": "E candidate, not selected",
    "E_pooled_s20_fb3": "E candidate, not selected",
    "E_pooled_s20_fb3_w95": "E candidate, not selected",
}


def boot_ci(v, reps=4000, seed=20261122):
    v = np.asarray(v, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(reps, len(v)))
    m = v[idx].mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main():
    frames = []
    for tag, ms in TAGS.items():
        p = RES / tag / "trials.csv"
        d = pd.read_csv(p)
        frames.append(d[d.method.isin(ms)][["case", "seed", "alpha", "beta", "method",
                                            "weighted_on_time_rate", "weighted_completion_rate",
                                            "cost", "observations", "invalid_actions", "independent_diff"]])
    d = pd.concat(frames, ignore_index=True)
    d = d.drop_duplicates(["method", "case", "seed", "alpha", "beta"])
    d.to_csv(RES / "trials.csv", index=False)

    base = d[d.method == "baseline"]
    hold = json.loads((RES / "holdout_G2" / "summary.json").read_text(encoding="utf8"))["by_fold"]
    for extra in sorted(RES.glob("holdout_*/summary.json")):
        if extra.parent.name == "holdout_G2":
            continue
        try:
            more = json.loads(extra.read_text(encoding="utf8")).get("by_fold", {})
        except Exception:
            continue
        for k, v in more.items():
            hold.setdefault(k, v)
    oracle = json.loads((RES / "oracle_dp" / "summary.json").read_text(encoding="utf8"))
    chain = {}
    cp = RES / "chain_probe" / "trials.csv"
    if cp.exists():
        c = pd.read_csv(cp)
        chain = {m: float(g.abs_diff.max()) for m, g in c.groupby("method")}

    out = {"protocol": {
        "cases": ["S1", "S2", "S3"], "seeds": int(d.seed.nunique()), "mappings": MAPPINGS,
        "hidden_sampling": "verbatim confirm_lookahead.hidden_for (random.Random(seed), jobs_for order, validation&eligible pool)",
        "paired": "baseline and each method share the same hidden draw per (case,seed)",
        "rescore": "engine.independent_metrics; tolerance 1e-9",
        "bootstrap": {"seed": 20261122, "reps": 4000, "cluster": "seed (nominal, mean over cases)"},
        "baseline_nominal_pct": float(base[(base.alpha == 1) & (base.beta == .5)].weighted_on_time_rate.mean() * 100),
    }, "oracle_dp_verified": oracle, "chain_gating_probe_max_abs_diff": chain,
        "methods": {}, "baseline": {}}

    def stats(m):
        n = d[(d.method == m) & (d.alpha == NOM[0]) & (d.beta == NOM[1])]
        n_total = int((d.method == m).sum())
        nominal_only = n_total < 450
        by_case = n.groupby("case").weighted_on_time_rate.mean()
        base_nom = base[(base.alpha == NOM[0]) & (base.beta == NOM[1])]
        pairs = n.set_index(["case", "seed"]).weighted_on_time_rate.to_frame("m").join(
            base_nom.set_index(["case", "seed"]).weighted_on_time_rate.to_frame("b"), how="inner")
        delta = (pairs.m - pairs.b) * 100
        by_seed = delta.groupby(level="seed").mean()
        lo, hi = boot_ci(by_seed.values)
        per_map = {}
        for a, b in MAPPINGS:
            x = d[(d.method == m) & (d.alpha == a) & (d.beta == b)].weighted_on_time_rate
            y = base[(base.alpha == a) & (base.beta == b)].weighted_on_time_rate
            if len(x) == 0:
                per_map[f"{a}/{b}"] = None
            else:
                per_map[f"{a}/{b}"] = dict(mean=float(x.mean()), baseline=float(y.mean()),
                                           delta_pp=float((x.mean() - y.mean()) * 100))
        return dict(
            nominal_pct=float(by_case.mean() * 100), by_case_pct={k: float(v * 100) for k, v in by_case.items()},
            five_mapping_pct=None if nominal_only else float(d[d.method == m].weighted_on_time_rate.mean() * 100),
            per_mapping=per_map,
            delta_pp=float(delta.mean()), delta_cases_pp={k: float(v) for k, v in ((pairs.m - pairs.b) * 100).groupby(level="case").mean().items()},
            ci95_pp=[lo, hi], positive_seeds=int((by_seed > 0).sum()), n_seeds=int(len(by_seed)),
            cost=float(n.cost.mean()), completion=float(n.weighted_completion_rate.mean()),
            observations=float(n.observations.mean()),
            max_independent_diff=float(n.independent_diff.max()), invalid_actions=int(n.invalid_actions.sum()),
            n_runs=int(len(n)), nominal_only=nominal_only,
            selection=SELECTION.get(m, ""),
            holdout_pp=None if m not in hold else dict(mean=hold[m]["mean_pp"], fold0=hold[m]["fold0_pp"],
                                                       fold1=hold[m]["fold1_pp"]),
        )

    out["baseline"] = stats("baseline")
    for m in [x for x in d.method.unique() if x != "baseline"]:
        s = stats(m)
        checks = {
            "nominal_ge_90.198": (s["nominal_pct"] >= 90.198) if not s["nominal_only"] else False,
            "paired_ci_lower_gt_0": s["ci95_pp"][0] > 0,
            "all_mappings_ge_baseline": (not s["nominal_only"]) and all(
                v is not None and v["delta_pp"] >= -1e-9 for v in s["per_mapping"].values()),
            "holdout_mean_nonneg": None if s["holdout_pp"] is None else s["holdout_pp"]["mean"] >= 0,
            "independent_diff_lt_1e-9": s["max_independent_diff"] < 1e-9,
            "no_invalid_actions": s["invalid_actions"] == 0,
        }
        s["gate_checks"] = checks
        s["gate_pass"] = all(v is True for v in checks.values())
        out["methods"][m] = s

    out["paper_ready"] = make_paragraph(out)
    n = d[(d.alpha == NOM[0]) & (d.beta == NOM[1])]
    piv = n.pivot_table(index=["case", "seed"], columns="method",
                        values="weighted_on_time_rate")
    if {"official_s40", "audit_ref"} <= set(piv.columns):
        dd = (piv["official_s40"] - piv["audit_ref"]).dropna() * 100
        by_seed = dd.groupby(level="seed").mean()
        out["audit_ab"] = {
            "official_s40_nominal_pct": float(piv["official_s40"].mean() * 100),
            "audit_ref_nominal_pct": float(piv["audit_ref"].mean() * 100),
            "official_minus_corrected_pp": float(by_seed.mean()),
            "boot95_pp": list(boot_ci(by_seed.values)),
            "positive_seeds": int((by_seed > 0).sum()),
            "n_seeds": int(len(by_seed)),
            "note": "同一 harness、同种子配对；official 含 P1 时钟/链式/P2 预算缺陷，修正后降幅即缺陷带来的虚高",
        }
    out["p1p2_audit"] = {
        "A_多步前瞻": {"clock_end_is_start_plus_p": True, "chain_probability_effective": False,
                    "budget_screen_before_rank": True, "grid_is_discrete_disclosed": True,
                    "note": "祖先门控为代码级空操作：把 _ancestors 置空后 9 个 episode 逐位相同；后代仍被当作必然按时完成"},
        "B_联合场景前瞻": {"clock_end_is_start_plus_p": True, "chain_probability_effective": True,
                     "budget_screen_before_rank": True, "grid_is_discrete_disclosed": True,
                     "note": "场景内 C>H 即不释放后继，链式失败被真实模拟；闭包用均值时长作点估计，属启发式而非概率链"},
        "C_尾部价值DP": {"clock_end_is_start_plus_p": True, "chain_probability_effective": True,
                     "budget_screen_before_rank": True, "grid_is_discrete_disclosed": True,
                     "note": "DP 用 need/done 掩码强制前驱完成；时间网格 dt 与分位场景均明确声明为离散近似"},
        "D_信息价值": {"clock_end_is_start_plus_p": True, "chain_probability_effective": False,
                   "budget_screen_before_rank": True, "grid_is_discrete_disclosed": True,
                   "note": "与 A 同源，祖先门控同样为空操作；但 D 的结论是不启用观测，不影响其否定性结论"},
        "E_稳健分布与选参": {"clock_end_is_start_plus_p": True, "chain_probability_effective": False,
                     "budget_screen_before_rank": True, "grid_is_discrete_disclosed": True,
                     "note": "祖先门控为空操作（探针差 0.0）；实测增益应归因于环境平衡的稳健时长先验与强度 40，而非链式概率"},
        "F_上界与可行性": {"note": "原始 trials_S1/S2/S3（各 600 行）+ legal_extra（450 行）完整；运行器使用 solve_oracle_dp（子集 DP），非 solve_oracle（MILP）；本审计独立复算 DP 上界 95.73%，对全部策略 0 违例；relaxation_ub 98.45% 亦 0 违例。尚未产出顶层 trials.csv/summary.json 与 report.md，属交付缺口。solve_oracle(MILP) 在 status=0 时仍被在线策略超越，属未使用的不健全实现，不作为上界引用"},
    }
    out["recompute_discrepancies"] = {
        "all_core_point_estimates_match": True,
        "max_abs_point_difference_pp_vs_roles": 0.0,
        "max_bootstrap_ci_endpoint_difference_pp": 0.06,
        "details": [
            "A：标称 90.243/五映射 87.618 与本单元复算逐位一致；其留出只用 12 种子，本单元按 holdout_lookahead.py 用 20 种子，符号相反（A 报告 −0.45pp / 本单元 +0.84pp），A 的留出标为协议不一致、不可采信",
            "B：K32/K64/K128 标称与五映射、以及 +0.59/+1.38 留出与本单元逐位一致",
            "C：角色 C 已补齐顶层 trials.csv（2700 行，30 种子×5 映射×6 方法）与 report.md；本单元独立重跑其提交配置 dp_mc_t4_s256_dt10，标称 91.088、五映射 88.764、配对 +6.00 [4.57,7.34]、27/30，与 C 自报逐位一致（CI 端点差 <0.04pp，实现差异）",
            "D：四臂标称 90.243/88.056/90.104/87.870 与本单元逐位一致；D 已自行撤回不可复现的 prefix 面板",
            "E：90.451/88.574、留出 +1.649 与本单元逐位一致；仅 bootstrap 实现不同，CI 端点差 <0.06pp",
            "F：角色 F 已补齐顶层 trials.csv（2250 行）与 report.md；本单元复算 oracle 95.729（S1 99.375 / S2 95.000 / S3 92.813）、ub_relax 98.449、v2 90.243、pressure_gated 85.093 逐位一致。另：其 lookahead_s40 在本单元复现为五映射 87.5185，与 F 自报一致；该配置是未修正的官方实现（含 P1/P2 缺陷），修正后 audit_ref 为 89.988 / 87.120",
            "共享参考 v2_s20 90.16%：来自 C 的 v2_reference 副本；A/E 的 multi_step_policy v2 为 90.243%，差 0.083pp，属实现差异而非协议差异，本报告只用自己复算的 90.243%",
            "共享参考审计修正版 89.9884% 与五映射 87.12%：本单元复算逐位一致",
        ],
    }
    out["ranking"] = [
        {"rank": 1, "method": "E_env_s40_fb3", "role": "稳健候选", "why": "唯一在 train 环境分组 CV 上选参、验证集未参与；六项门槛全过；留出两折均非负；机制简单"},
        {"rank": 2, "method": "C_mc_t4_s256_dt10", "role": "激进候选", "why": "标称 91.088 为已核验最高的提交配置，五映射 88.764，留出 +2.18；但配置为其候选表在同一 validation 上的事后最大，含选择偏差，且留出 fold0 −0.59"},
        {"rank": 3, "method": "B_K64", "role": "激进备选", "why": "五映射最高 88.961、标称 90.961；K 按 max-K 规则在同一 validation 上选定，留出 +0.59 且 fold0 −0.63"},
        {"rank": 4, "method": "B_K128", "role": "不采用为主", "why": "标称最高 91.169，但 K=128 超出预注册网格，五映射 88.458 低于 K64"},
        {"rank": 5, "method": "A_v2", "role": "不采用", "why": "标称仅超门槛 0.045pp；留出符号随种子集翻转；祖先链门控为空操作"},
    ]
    out["adoption"] = {
        "single": "E_env_s40_fb3",
        "aggressive": "C_mc_t4_s256_dt10", "aggressive_alt": "B_K64",
        "robust": "E_env_s40_fb3",
        "caveat": "两候选均只在本验证协议与当前数据上达到门槛；不得写成跨环境泛化突破",
    }
    out["code_snapshot_sha256_16"] = _code_sha()
    out["deliverable_gaps"] = {
        "generated_from": "filesystem scan of parallel/<role>/ (report.md, results/trials.csv, results/summary.json)",
        "complete": ["B_联合场景前瞻", "C_尾部价值DP", "D_信息价值", "E_稳健分布与选参", "F_上界与可行性"],
        "incomplete": {
            "A_多步前瞻": {
                "missing": ["results/trials.csv", "results/summary.json"],
                "report_placeholders": 4,
                "available_data": "results/full 1350 行(30种子×5映射×baseline/v2/d2b4q5)；results/ref30 1350 行(baseline/v2/audit)；results/holdout 216 行(12种子)",
                "issue": "无顶层规范产物；holdout 仅 12 种子，与 holdout_lookahead.py 的 20 种子协议不一致；report 尚有 4 处 TBD（第 111/151/155/177 行）",
                "fix": "在 A 目录汇总 results/full 为顶层 trials.csv/summary.json；报告 4 处 TBD 改为实测值；holdout 改用 20 种子重跑",
            },
        },
        "trust": {
            "A": "点估计可采信（与 G 复算逐位一致）；留出数字不可采信（12 种子）",
            "C": "已补齐并核验：提交配置 dp_mc_t4_s256_dt10 由 G 独立重跑一致",
            "F": "已补齐并核验：oracle/ub/v2/基线由 G 独立重跑一致",
        },
        "note": "本单元遵守“不得改动其它目录”，未替 A 生成文件；缺口清单仅作交接。",
    }
    (RES / "summary.json").write_text(json.dumps(out, indent=2, ensure_ascii=False, default=clean), encoding="utf8")

    # markdown table
    hdr = "| method | nominal% | five-map% | paired Δpp [95% CI] | +seeds | holdout Δpp (f0/f1) | invalid | max|ind| | gate |"
    print(hdr); print("|" + "---|" * 9)
    order = sorted(out["methods"].items(), key=lambda kv: -kv[1]["nominal_pct"])
    for m, s in order:
        h = s["holdout_pp"]
        hs = "—" if h is None else f'{h["mean"]:+.2f} ({h["fold0"]:+.2f}/{h["fold1"]:+.2f})'
        fm = "—" if s["five_mapping_pct"] is None else f'{s["five_mapping_pct"]:.3f}'
        print(f'| {m} | {s["nominal_pct"]:.3f} | {fm} | '
              f'{s["delta_pp"]:+.2f} [{s["ci95_pp"][0]:.2f},{s["ci95_pp"][1]:.2f}] | '
              f'{s["positive_seeds"]}/{s["n_seeds"]} | {hs} | {s["invalid_actions"]} | '
              f'{s["max_independent_diff"]:.1e} | {"PASS" if s["gate_pass"] else "—"} |')


def clean(o):
    if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
        return None
    raise TypeError


def _code_sha():
    import hashlib
    par = ROLE.parent
    files = {
        "A": "A_多步前瞻/code/multi_step_policy.py",
        "B": "B_联合场景前瞻/code/joint_policy.py",
        "C": "C_尾部价值DP/code/dp_tail_policy.py",
        "D": "D_信息价值/code/info_policy.py",
        "E": "E_稳健分布与选参/code/robust_chain_policy.py",
        "E_configs": "E_稳健分布与选参/code/configs.py",
        "F": "F_上界与可行性/code/oracle.py",
    }
    out = {}
    for k, v in files.items():
        p = par / v
        if p.exists():
            out[k] = hashlib.sha256(p.read_bytes()).hexdigest()[:16]
    return out


def make_paragraph(out):
    b = out["baseline"]
    e = out["methods"]["E_env_s40_fb3"]
    bb = out["methods"]["B_K64"]
    cc = out["methods"]["C_mc_t4_s256_dt10"]
    return (
        "在统一的 3 案例 × 30 个验证种子 × 5 种映射配对协议（隐藏时长逐字复用 confirm_lookahead.hidden_for，"
        f"正式 Engine 计分并以 independent_metrics 复核）下，压力门控基线标称按时率为 {b['nominal_pct']:.2f}%"
        f"（五映射均值 {b['five_mapping_pct']:.2f}%）。按训练环境分组交叉验证预先选定、验证集未参与选参的"
        f"稳健先验策略 env_s40_fb3 取得标称 {e['nominal_pct']:.2f}%（配对提升 {e['delta_pp']:+.2f}pp，"
        f"按种子 bootstrap 95% 区间 [{e['ci95_pp'][0]:.2f}, {e['ci95_pp'][1]:.2f}]，"
        f"30 个种子中 {e['positive_seeds']} 个为正），五映射均值 {e['five_mapping_pct']:.2f}% 且五种映射逐一不低于基线；"
        f"环境分组留出两组均值 {e['holdout_pp']['mean']:+.2f}pp"
        f"（两折 {e['holdout_pp']['fold0']:+.2f} / {e['holdout_pp']['fold1']:+.2f}pp，均为非负），"
        f"独立计分最大差 {e['max_independent_diff']:.1e}，非法动作 {e['invalid_actions']} 个。"
        f"在验证集上标称更高的联合场景前瞻（K=64，{bb['nominal_pct']:.2f}%）与尾部蒙特卡洛 DP"
        f"（{cc['nominal_pct']:.2f}%）分别在 K 与候选参数上做过事后选择，其点估计含选择偏差"
        f"（环境留出仅 {bb['holdout_pp']['mean']:+.2f} 与 {cc['holdout_pp']['mean']:+.2f}pp，且各自一折为负），"
        "故只作为探索性上探，不作为无偏效果。"
        "此外，未修正的官方前瞻基准（s40）在同一框架下复现为标称 90.80%、五映射 87.52%，与角色 F 自报一致；"
        "打开时钟、链式与预算三项审计修正后降为 89.99% / 87.12%，两者标称差 +0.81pp"
        "（配对 95% 区间 [−0.31, 2.11]，16/30 为正，不显著），但分别落在 6% 门槛两侧。"
        "因此“某方法是否过线”对尾部实现的正确性敏感，不能直接引用未修正版本。"
        "需要强调三点：其一，环境分组留出下的基线均值（约 78.39%）明显低于验证集（85.09%），"
        "说明留出环境与验证环境存在系统性差异，留出增益的绝对幅度不能与验证集增益直接等同；"
        "其二，A/D/E 三者的“祖先链完成概率门控”经把 _ancestors 置空的探针对照后逐位不变（差 0.0），"
        "属代码级空操作，因此上述增益应归因于环境平衡的稳健时长先验与反馈强度，而非链式概率建模；"
        "其三，独立复算的完美信息子集 DP 最优为 95.73%（S1 99.375 / S2 95.000 / S3 92.813），"
        "对全部在线策略 0 违例，故 90.45% 距上界仍有约 5.3pp 空间，不存在“已逼近最优”的证据。"
        "综上，本报告结论为：稳健先验策略在给定协议与当前数据上达到 6% 相对目标的门槛，"
        "但证据强度分级为“验证集强、环境留出弱”，不得表述为已证实的跨环境泛化突破。"
    )


if __name__ == "__main__":
    main()
