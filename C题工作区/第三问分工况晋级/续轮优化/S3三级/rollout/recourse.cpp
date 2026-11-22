// Causal recourse policy.
//
// Every feasible first job is scored by common, independent hypothetical
// duration scenarios.  Inside a scenario the downstream dispatcher replans
// greedily from the public simulated state (time, location, completed set,
// remaining budget).  Its ordering decisions use only public information:
// releases, dues, weights, distances and the *typical* duration row.  The
// scenario's future durations are used only to realise completion times and
// reward, never to order the remaining jobs.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace {

constexpr double kEps = 1e-9;
constexpr double kReserve = 0.2;   // minimum occupancy between completions
constexpr double kPrep = 0.5;      // preparation before a service starts
constexpr double kFixed = 2.02;    // fixed fee per executed job
constexpr double kPerDist = 0.2;   // fee per unit distance
constexpr int kMaxWaits = 2;       // at most two waits between completions
constexpr int kRuleCount = 6;
constexpr int kMaxScenarios = 512;

inline bool good(double x) { return std::isfinite(x); }

inline int bit_count(std::uint32_t x) {
    int c = 0;
    while (x) {
        x &= x - 1u;
        ++c;
    }
    return c;
}

struct Ctx {
    int n = 0;
    int ns = 0;
    std::uint32_t all = 0;
    std::uint32_t done = 0;
    double now = 0.0;
    double budget = 0.0;
    double horizon = 0.0;
    const double* release = nullptr;
    const double* due = nullptr;
    const double* weight = nullptr;
    const double* dist = nullptr;
    const double* sc = nullptr;    // scenario durations, ns rows of n
    const double* typ = nullptr;   // typical durations (last services row)
    const int* parent = nullptr;   // prerequisite bitmask per job
    double cw = 0.0;
    double urgency = 1.0;
    double travel = 1.0;
    std::uint32_t descendants[16] = {};
    std::uint32_t valid_mask = 0;
    int parent_store[16] = {};
    double typ_store[16] = {};
    bool valid[16] = {};

    inline double leg(int from, int to) const {
        return dist[from * n + to];
    }
    inline double fee(int from, int to) const {
        return kFixed + kPerDist * leg(from, to);
    }
    inline bool prereq_ok(std::uint32_t mask, int j) const {
        return (static_cast<std::uint32_t>(parent[j]) & ~mask) == 0u;
    }
};

inline double slack_of(const Ctx& c, int cur, double t, int j) {
    return c.due[j] - (t + c.leg(cur, j) + kPrep + c.typ[j]);
}

// Higher is executed first.  Rules only see public information.
inline double rule_score(const Ctx& c, int rule, int cur, double t,
                         std::uint32_t mask, int j) {
    const double d = c.leg(cur, j);
    switch (rule) {
        case 0:  // earliest due date
            return -c.due[j];
        case 1: {  // weight density: value per occupied time plus travel
            const double denom = c.typ[j] + std::max(0.05, c.travel * d);
            return c.weight[j] / denom;
        }
        case 2:  // heaviest weight first
            return c.weight[j];
        case 3:  // least slack first
            return -slack_of(c, cur, t, j);
        case 4: {  // unlock the most remaining successors, then least slack
            const std::uint32_t rest =
                c.descendants[j] & c.valid_mask & ~mask & ~(1u << j);
            return static_cast<double>(bit_count(rest)) * 1e6 -
                   slack_of(c, cur, t, j);
        }
        default:  // nearest neighbour
            return -d;
    }
}

inline bool tie_prefer(const Ctx& c, int cur, int a, int b) {
    if (c.due[a] != c.due[b]) return c.due[a] < c.due[b];
    const double da = c.leg(cur, a);
    const double db = c.leg(cur, b);
    if (da != db) return da < db;
    return a < b;
}

inline bool feasible(const Ctx& c, std::uint32_t mask, double t,
                     double budget_left, int cur, int j) {
    if (j < 0 || j >= c.n || !c.valid[j]) return false;
    if (mask & (1u << j)) return false;
    if (!c.prereq_ok(mask, j)) return false;
    if (c.release[j] > t + kEps) return false;
    const double d = c.leg(cur, j);
    if (!good(d) || d < 0.0) return false;
    if (c.fee(cur, j) > budget_left + kEps) return false;
    const double start = t + d + kPrep;
    if (start > c.horizon + kEps) return false;
    if (start + std::max(kReserve, c.typ[j]) > c.horizon + kEps)
        return false;
    return true;
}

inline int choose_greedy(const Ctx& c, int rule, std::uint32_t mask, double t,
                         double budget_left, int cur) {
    int best = -1;
    double best_score = -std::numeric_limits<double>::infinity();
    for (int j = 0; j < c.n; ++j) {
        if (!feasible(c, mask, t, budget_left, cur, j)) continue;
        const double s = rule_score(c, rule, cur, t, mask, j);
        if (best < 0 || s > best_score + 1e-12) {
            best = j;
            best_score = s;
        } else if (std::fabs(s - best_score) <= 1e-12 &&
                   tie_prefer(c, cur, j, best)) {
            best = j;
            best_score = s;
        }
    }
    return best;
}

struct Sim {
    std::uint32_t mask = 0;
    double t = 0.0;
    double budget = 0.0;
    double total = 0.0;
    int loc = 0;
};

// Execute job j with the scenario duration; reward is public-state based.
inline bool execute(const Ctx& c, Sim& s, int j, int scenario,
                    double& reward_out) {
    const double d = c.leg(s.loc, j);
    const double start = s.t + d + kPrep;
    double dur = c.sc[static_cast<std::size_t>(scenario) * c.n + j];
    if (!good(dur) || dur < 0.0) dur = c.typ[j];

    s.budget -= c.fee(s.loc, j);
    s.mask |= (1u << j);
    s.loc = j;
    const double finish = start + dur;
    s.t = start + std::max(kReserve, dur);

    reward_out = 0.0;
    if (finish > c.horizon + kEps) return false;  // no completion in horizon

    const double ontime = (finish <= c.due[j] + kEps) ? 1.0 : 0.0;
    double r = c.weight[j] * (ontime + c.cw);
    double late = finish - c.due[j];
    if (late > 0.0) {
        if (late > 10.0) late = 10.0;
        const double u = 1.0 + std::clamp(c.urgency, 0.0, 4.0);
        r -= 0.0005 * c.weight[j] * late * u;  // small tardiness reduction
    }
    reward_out = r;
    s.total += r;
    return true;
}

double run_rollout(const Ctx& c, int rule, int scenario, std::uint32_t mask,
                   double t, int loc, double budget_left, int forced_first) {
    Sim s;
    s.mask = mask;
    s.t = t;
    s.budget = budget_left;
    s.total = 0.0;
    s.loc = loc;

    int forced = forced_first;
    int waits_in_gap = 0;
    while (true) {
        int j = -1;
        if (forced >= 0) {
            j = forced;
            forced = -1;
            if (!feasible(c, s.mask, s.t, s.budget, s.loc, j)) break;
        } else {
            j = choose_greedy(c, rule, s.mask, s.t, s.budget, s.loc);
        }

        if (j >= 0) {
            double r = 0.0;
            const bool completed = execute(c, s, j, scenario, r);
            if (!completed) break;
            waits_in_gap = 0;  // a completion closes the current idle gap
            continue;
        }
        if (s.mask == c.all) break;
        if (waits_in_gap >= kMaxWaits) break;

        // No ready feasible job: advance to the next release.
        double next_rel = std::numeric_limits<double>::infinity();
        for (int k = 0; k < c.n; ++k) {
            if (!c.valid[k] || (s.mask & (1u << k))) continue;
            const double rk = c.release[k];
            if (!good(rk)) continue;
            if (rk > s.t + kEps && rk <= c.horizon + kEps && rk < next_rel)
                next_rel = rk;
        }
        if (!(next_rel <= c.horizon + kEps)) break;
        s.t = next_rel;
        ++waits_in_gap;
    }
    return s.total;
}

int solve(const Ctx& c) {
    if (c.done == c.all) return -1;

    const int ns = c.ns;
    int best_job = -1;
    double best_value = -std::numeric_limits<double>::infinity();

    // For each feasible first job pick one fixed downstream rule by mean
    // return over the common scenarios, then compare the first jobs.
    for (int f = 0; f < c.n; ++f) {
        if (!feasible(c, c.done, c.now, c.budget, c.n, f)) continue;
        double action_value = -std::numeric_limits<double>::infinity();
        for (int rule = 0; rule < kRuleCount; ++rule) {
            double sum = 0.0;
            for (int s = 0; s < ns; ++s)
                sum += run_rollout(c, rule, s, c.done, c.now, c.n, c.budget, f);
            const double avg = sum / static_cast<double>(ns);
            if (avg > action_value) action_value = avg;
        }
        if (best_job < 0 || action_value > best_value + 1e-12 ||
            (std::fabs(action_value - best_value) <= 1e-12 &&
             tie_prefer(c, c.n, f, best_job))) {
            best_value = action_value;
            best_job = f;
        }
    }

    // Value of waiting for the next release, evaluated the same way.
    bool wait_ok = false;
    double wait_value = -std::numeric_limits<double>::infinity();
    double next_rel = std::numeric_limits<double>::infinity();
    for (int k = 0; k < c.n; ++k) {
        if (!c.valid[k] || (c.done & (1u << k))) continue;
        const double rk = c.release[k];
        if (!good(rk)) continue;
        if (rk > c.now + kEps && rk <= c.horizon + kEps && rk < next_rel)
            next_rel = rk;
    }
    if (next_rel <= c.horizon + kEps) {
        wait_ok = true;
        for (int rule = 0; rule < kRuleCount; ++rule) {
            double sum = 0.0;
            for (int s = 0; s < ns; ++s)
                sum += run_rollout(c, rule, s, c.done, next_rel, c.n, c.budget,
                                   -1);
            const double avg = sum / static_cast<double>(ns);
            if (avg > wait_value) wait_value = avg;
        }
    }

    const double tol = 1e-9 * (1.0 + std::fabs(best_value));
    if (best_job < 0) {
        if (wait_ok && wait_value > 1e-12) return -2;
        return -1;
    }
    if (wait_ok && wait_value > best_value + tol) return -2;
    if (!(best_value > 1e-12)) return -1;
    return best_job;
}

bool init(Ctx& c, int n, int done, int ns, double now, double budget,
          double horizon, const double* release, const double* due,
          const double* weights, const double* distances,
          const double* services, const int* parents, double cw,
          double urgency, double travel) {
    if (n < 1 || n > 16) return false;
    if (!release || !due || !weights || !distances || !services || !parents)
        return false;
    if (!good(now) || !good(budget) || !good(horizon) || !good(cw))
        return false;

    c.n = n;
    c.all = (n == 16) ? 0xFFFFu : ((1u << n) - 1u);
    c.done = static_cast<std::uint32_t>(done) & c.all;
    c.now = now;
    c.budget = budget;
    c.horizon = horizon;
    c.release = release;
    c.due = due;
    c.weight = weights;
    c.dist = distances;
    c.cw = cw;
    c.urgency = good(urgency) ? urgency : 1.0;
    c.travel = good(travel) ? travel : 1.0;

    int raw_ns = ns;
    if (raw_ns < 0) raw_ns = 0;
    int scen = raw_ns;
    if (scen > kMaxScenarios) scen = kMaxScenarios;
    c.ns = (scen > 0) ? scen : 1;
    c.sc = services;
    const double* raw_typ =
        services + static_cast<std::size_t>(raw_ns) * static_cast<std::size_t>(n);

    for (int j = 0; j < n; ++j) {
        c.valid[j] = good(release[j]) && good(due[j]) && good(weights[j]);
        c.parent_store[j] =
            static_cast<int>(static_cast<std::uint32_t>(parents[j]) & c.all);
        if (c.parent_store[j] & (1 << j)) c.valid[j] = false;
        c.descendants[j] = 0u;

        double tval = raw_typ[j];
        if (!good(tval) || tval < 0.0) {
            double sum = 0.0;
            int count = 0;
            for (int s = 0; s < c.ns; ++s) {
                const double v =
                    c.sc[static_cast<std::size_t>(s) * n + j];
                if (good(v) && v >= 0.0) {
                    sum += v;
                    ++count;
                }
            }
            tval = count > 0 ? sum / static_cast<double>(count) : -1.0;
        }
        if (!good(tval) || tval < 0.0) c.valid[j] = false;
        c.typ_store[j] = (tval >= 0.0) ? tval : 0.0;
    }
    c.valid_mask = 0u;
    for (int j = 0; j < n; ++j) {
        if (c.valid[j]) c.valid_mask |= (1u << j);
    }
    c.parent = c.parent_store;
    c.typ = c.typ_store;

    // Transitive successors, independent of every scenario.
    for (int iter = 0; iter < n; ++iter) {
        bool changed = false;
        for (int k = 0; k < n; ++k) {
            const std::uint32_t pm =
                static_cast<std::uint32_t>(c.parent_store[k]);
            if (!pm) continue;
            const std::uint32_t add = (1u << k) | c.descendants[k];
            for (int j = 0; j < n; ++j) {
                if (!(pm & (1u << j))) continue;
                if ((c.descendants[j] & add) != add) {
                    c.descendants[j] |= add;
                    changed = true;
                }
            }
        }
        if (!changed) break;
    }
    return true;
}

}  // namespace

extern "C" __declspec(dllexport) int recourse(
    int n, int done, int ns, double now, double budget, double horizon,
    const double* release, const double* due, const double* weights,
    const double* distances, const double* services, const int* parents,
    double cw, double urgency, double travel) {
    Ctx c;
    if (!init(c, n, done, ns, now, budget, horizon, release, due, weights,
              distances, services, parents, cw, urgency, travel))
        return -1;
    return solve(c);
}
