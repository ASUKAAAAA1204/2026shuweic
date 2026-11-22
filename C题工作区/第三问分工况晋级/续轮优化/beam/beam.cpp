#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace {

constexpr double kEps = 1e-9;
constexpr double kReserve = 0.2;
constexpr double kPrep = 0.5;
constexpr double kFixedCost = 2.02;

struct State {
    std::uint32_t mask;
    double now;
    double left;
    double reward;
    double bound;
    int loc;
    int first;
    int id;
};

inline int bit_count(std::uint32_t x) {
    int c = 0;
    while (x) {
        x &= x - 1u;
        ++c;
    }
    return c;
}

inline double on_time_value(double finish, double due_time, double margin) {
    if (margin > 0.0) {
        const double x = (finish - due_time) / margin;
        if (x <= -1.0) return 1.0;
        if (x >= 1.0) return 0.0;
        const double u = 0.5 * (1.0 - x);
        return u * u * (3.0 - 2.0 * u);
    }
    return finish <= due_time + kEps ? 1.0 : 0.0;
}

struct Planner {
    int n;
    std::uint32_t all_mask;
    std::uint32_t root_mask;
    double root_now;
    double root_left;
    double horizon;
    double cw;
    double margin;
    const double* release;
    const double* due;
    const double* weight;
    const double* dist;
    const double* dur;
    int parent[16];
    double dmin[16];
    bool valid[16];
    int next_id;

    bool group_better(const State& a, const State& b) const {
        if (a.bound != b.bound) return a.bound > b.bound;
        if (a.reward != b.reward) return a.reward > b.reward;
        if (a.now != b.now) return a.now < b.now;
        if (a.left != b.left) return a.left > b.left;
        const int fa = (a.first >= 0 ? 0 : 1);
        const int fb = (b.first >= 0 ? 0 : 1);
        if (fa != fb) return fa < fb;
        return a.id < b.id;
    }

    bool best_better(const State& a, const State& b) const {
        if (a.reward != b.reward) return a.reward > b.reward;
        const int ca = bit_count(a.mask);
        const int cb = bit_count(b.mask);
        if (ca != cb) return ca > cb;
        const int fa = (a.first >= 0 ? 0 : 1);
        const int fb = (b.first >= 0 ? 0 : 1);
        if (fa != fb) return fa < fb;
        if (a.now != b.now) return a.now < b.now;
        if (a.left != b.left) return a.left > b.left;
        return a.id < b.id;
    }

    double optimistic(const State& s) const {
        const std::uint32_t remaining = all_mask & ~s.mask;
        if (!remaining || s.now > horizon + kEps || s.left < kFixedCost - kEps)
            return 0.0;

        int cap = n;
        const int budget_cap =
            static_cast<int>(std::floor((s.left + kEps) / kFixedCost));
        if (budget_cap < cap) cap = budget_cap;
        const int time_cap = static_cast<int>(
            std::floor((horizon - s.now + kEps) / (kPrep + kReserve)));
        if (time_cap < cap) cap = time_cap;
        if (cap <= 0) return 0.0;

        double values[16];
        int count = 0;
        for (int j = 0; j < n; ++j) {
            if (!(remaining & (1u << j)) || !valid[j]) continue;
            const double earliest =
                (s.now > release[j] ? s.now : release[j]) + dmin[j] + kPrep;
            const double occupy = std::max(kReserve, dur[j]);
            if (!(earliest + occupy <= horizon + kEps)) continue;
            const double wj = weight[j];
            const double per = (wj >= 0.0) ? wj * (1.0 + cw) : wj * cw;
            if (per > 0.0) values[count++] = per;
        }
        if (count == 0) return 0.0;
        std::sort(values, values + count,
                  [](double a, double b) { return a > b; });
        if (count > cap) count = cap;
        double sum = 0.0;
        for (int i = 0; i < count; ++i) sum += values[i];
        return sum;
    }

    void add_job(const State& s, int j, std::vector<State>& out) {
        if (!valid[j] || (s.mask & (1u << j))) return;
        if (release[j] > s.now + kEps) return;
        if ((static_cast<std::uint32_t>(parent[j]) & ~s.mask) != 0u) return;

        const double d = dist[s.loc * n + j];
        if (!std::isfinite(d) || d < 0.0) return;

        const double start = s.now + d + kPrep;
        const double service = dur[j];
        const double next = start + std::max(kReserve, service);
        if (!(next <= horizon + kEps)) return;

        const double cost = kFixedCost + 0.2 * d;
        if (cost > s.left + kEps) return;

        const double finish = start + service;
        double reward =
            weight[j] * (on_time_value(finish, due[j], margin) + cw);
        double late = finish - due[j];
        if (late > 0.0) {
            if (late > 10.0) late = 10.0;
            reward -= 0.0005 * weight[j] * late;
        }

        State child;
        child.mask = s.mask | (1u << j);
        child.now = next;
        child.left = s.left - cost;
        child.reward = s.reward + reward;
        child.loc = j;
        child.first = (s.first == -1 ? j : s.first);
        child.bound = child.reward + optimistic(child);
        child.id = next_id++;
        out.push_back(child);
    }

    void add_wait(const State& s, std::vector<State>& out) {
        if (s.mask == all_mask) return;
        double next = std::numeric_limits<double>::infinity();
        for (int j = 0; j < n; ++j) {
            if (!valid[j] || (s.mask & (1u << j))) continue;
            if ((static_cast<std::uint32_t>(parent[j]) & ~s.mask) != 0u) continue;
            const double r = release[j];
            if (r > s.now + kEps && r <= horizon + kEps && r < next) next = r;
        }
        if (!(next <= horizon + kEps)) return;
        State child = s;
        child.now = next;
        if (s.first == -1) child.first = -2;
        child.bound = child.reward + optimistic(child);
        child.id = next_id++;
        out.push_back(child);
    }

    int solve(int width) {
        if (width < 1) width = 1;
        if (width > 256) width = 256;

        State root;
        root.mask = root_mask;
        root.now = root_now;
        root.left = root_left;
        root.reward = 0.0;
        root.loc = n;
        root.first = -1;
        root.id = next_id++;
        root.bound = optimistic(root);

        std::vector<State> beam;
        std::vector<State> children;
        std::vector<State> unique;
        std::vector<State> kept;
        const std::size_t child_cap =
            static_cast<std::size_t>(width) * static_cast<std::size_t>(n + 1);
        beam.reserve(static_cast<std::size_t>(width));
        children.reserve(child_cap);
        unique.reserve(child_cap);
        kept.reserve(static_cast<std::size_t>(width));
        beam.push_back(root);

        bool have_best = false;
        bool have_job_best = false;
        State best;
        State job_best;
        const int rounds = 2 * n + 4;

        for (int round = 0; round < rounds && !beam.empty(); ++round) {
            children.clear();
            for (const State& s : beam) {
                if (s.mask == all_mask) continue;
                for (int j = 0; j < n; ++j) {
                    if (s.mask & (1u << j)) continue;
                    add_job(s, j, children);
                }
                add_wait(s, children);
            }
            if (children.empty()) break;

            for (const State& c : children) {
                if (!have_best || best_better(c, best)) {
                    best = c;
                    have_best = true;
                }
                if (c.first >= 0 &&
                    (!have_job_best || best_better(c, job_best))) {
                    job_best = c;
                    have_job_best = true;
                }
            }

            std::sort(children.begin(), children.end(),
                      [this](const State& a, const State& b) {
                          if (a.mask != b.mask) return a.mask < b.mask;
                          if (a.loc != b.loc) return a.loc < b.loc;
                          return group_better(a, b);
                      });

            unique.clear();
            for (std::size_t i = 0; i < children.size();) {
                std::size_t k = i + 1;
                while (k < children.size() &&
                       children[k].mask == children[i].mask &&
                       children[k].loc == children[i].loc) {
                    ++k;
                }
                unique.push_back(children[i]);
                i = k;
            }

            kept.clear();
            for (const State& c : unique) {
                if (c.bound + kEps >= best.reward) kept.push_back(c);
            }
            std::sort(kept.begin(), kept.end(),
                      [this](const State& a, const State& b) {
                          return group_better(a, b);
                      });
            if (static_cast<int>(kept.size()) > width)
                kept.resize(static_cast<std::size_t>(width));
            beam.swap(kept);
        }

        if (!have_best || !(best.reward > 1e-12)) return -1;
        if (best.first == -2) {
            // Waiting is only reported when it is genuinely preferable.
            if (have_job_best) {
                const double tolerance =
                    1e-11 * (1.0 + std::fabs(best.reward));
                if (job_best.reward + tolerance >= best.reward)
                    return job_best.first;
            }
            return -2;
        }
        return best.first >= 0 ? best.first : -1;
    }

    bool init(int n_, int done, double now, double left, double horizon_,
              const double* release_, const double* due_,
              const double* weight_, const double* dist_, const double* dur_,
              const int* parents_, double cw_, double margin_) {
        if (n_ < 1 || n_ > 16) return false;
        if (!release_ || !due_ || !weight_ || !dist_ || !dur_ || !parents_)
            return false;
        if (!std::isfinite(now) || !std::isfinite(left) ||
            !std::isfinite(horizon_) || !std::isfinite(cw_) ||
            !std::isfinite(margin_)) {
            return false;
        }

        n = n_;
        all_mask = (n == 16) ? 0xFFFFu : ((1u << n) - 1u);
        root_mask = static_cast<std::uint32_t>(done) & all_mask;
        root_now = now;
        root_left = left;
        horizon = horizon_;
        cw = cw_;
        margin = margin_;
        release = release_;
        due = due_;
        weight = weight_;
        dist = dist_;
        dur = dur_;
        next_id = 0;

        for (int j = 0; j < n; ++j) {
            valid[j] = std::isfinite(release[j]) && std::isfinite(due[j]) &&
                       std::isfinite(weight[j]) && std::isfinite(dur[j]) &&
                       dur[j] >= 0.0;
            parent[j] = static_cast<int>(
                static_cast<std::uint32_t>(parents_[j]) & all_mask);
            if (valid[j] && (parent[j] & (1 << j))) valid[j] = false;

            dmin[j] = std::numeric_limits<double>::infinity();
            if (!valid[j]) continue;
            for (int k = 0; k <= n; ++k) {
                if (k == j) continue;
                const double d = dist[k * n + j];
                if (std::isfinite(d) && d >= 0.0 && d < dmin[j]) dmin[j] = d;
            }
            if (!std::isfinite(dmin[j])) valid[j] = false;
        }
        return true;
    }
};

}  // namespace

extern "C" __declspec(dllexport) int beam(
    int n, int done, double now, double left, double H,
    const double* release, const double* due, const double* w,
    const double* distances, const double* dur, const int* parents,
    int width, double cw, double due_margin) {
    Planner planner;
    if (!planner.init(n, done, now, left, H, release, due, w, distances,
                      dur, parents, cw, due_margin)) {
        return -1;
    }
    return planner.solve(width);
}
