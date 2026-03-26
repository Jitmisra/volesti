// volesti — timing benchmark: sparse vs dense boundary oracle
//
// Measures wall-clock time of trigonometric_positive_intersect on the same
// poset represented as OrderPolytope (our O(n+|E|) oracle) vs HPolytope
// (the generic O(mn) oracle). Prints per-call microseconds and speedup.
//
// This is the empirical answer to the question: "is the dense mat-vec
// actually the bottleneck?"
//
// Usage: ./benchmark_oracle

#include <iostream>
#include <iomanip>
#include <chrono>
#include <random>
#include <vector>
#include <cmath>

#include "Eigen/Eigen"
#include "cartesian_geom/cartesian_kernel.h"
#include "cartesian_geom/point.h"
#include "convex_bodies/hpolytope.h"
#include "convex_bodies/orderpolytope.h"
#include "misc/poset.h"

typedef double NT;
typedef Cartesian<NT> Kernel;
typedef typename Kernel::Point Point;
typedef HPolytope<Point> HP;
typedef OrderPolytope<Point> OP;
typedef Eigen::Matrix<NT, Eigen::Dynamic, Eigen::Dynamic> MT;
typedef Eigen::Matrix<NT, Eigen::Dynamic, 1> VT;


// Generate a sparse random DAG: for elements 0..n-1, add edge (i,j) with
// i < j with probability p, then take the transitive reduction (we just
// keep direct edges, since OrderPolytope handles transitivity).
Poset make_sparse_poset(unsigned n, double p, unsigned seed = 42) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> coin(0.0, 1.0);
    Poset::RV rels;
    for (unsigned i = 0; i < n; ++i)
        for (unsigned j = i + 1; j < n; ++j)
            if (coin(gen) < p)
                rels.push_back({i, j});
    return Poset(n, rels);
}


struct BenchResult {
    double us_per_call_op;   // microseconds per call, OrderPolytope
    double us_per_call_hp;   // microseconds per call, HPolytope
    double speedup;
    unsigned n;
    unsigned edges;
    unsigned hyperplanes;
};


BenchResult run_benchmark(unsigned n, double edge_prob, unsigned num_calls = 5000) {
    Poset poset = make_sparse_poset(n, edge_prob);

    OP order_poly(poset);
    MT A = order_poly.get_dense_mat();
    VT b = order_poly.get_vec();
    HP h_poly(n, A, b);

    VT inner = order_poly.inner_point();

    std::mt19937 gen(7);
    std::normal_distribution<NT> normal(0.0, 1.0);
    std::uniform_real_distribution<NT> unif(0.5, 3.0);

    // Pre-generate all random inputs so we measure only the oracle
    struct OracleInput {
        Point pos;
        Point vel;
        NT omega;
    };
    std::vector<OracleInput> inputs(num_calls);
    for (unsigned i = 0; i < num_calls; ++i) {
        VT pos = inner;
        for (unsigned j = 0; j < n; ++j)
            pos(j) += normal(gen) * 0.005;
        for (unsigned j = 0; j < n; ++j)
            pos(j) = std::max(0.01, std::min(0.99, pos(j)));

        VT vel(n);
        for (unsigned j = 0; j < n; ++j)
            vel(j) = normal(gen);

        inputs[i] = {Point(pos), Point(vel), unif(gen)};
    }

    // ---- Benchmark OrderPolytope (sparse) ----
    volatile NT sink_t = 0;
    auto t0 = std::chrono::high_resolution_clock::now();
    for (unsigned i = 0; i < num_calls; ++i) {
        int fp = -1;
        auto [t_hit, facet] = order_poly.trigonometric_positive_intersect(
            inputs[i].pos, inputs[i].vel, inputs[i].omega, fp);
        sink_t += t_hit;  // prevent optimizer from removing the call
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double us_op = std::chrono::duration<double, std::micro>(t1 - t0).count() / num_calls;

    // ---- Benchmark HPolytope (dense) ----
    auto t2 = std::chrono::high_resolution_clock::now();
    for (unsigned i = 0; i < num_calls; ++i) {
        int fp = -1;
        auto [t_hit, facet] = h_poly.trigonometric_positive_intersect(
            inputs[i].pos, inputs[i].vel, inputs[i].omega, fp);
        sink_t += t_hit;
    }
    auto t3 = std::chrono::high_resolution_clock::now();
    double us_hp = std::chrono::duration<double, std::micro>(t3 - t2).count() / num_calls;

    // prevent dead code elimination
    if (sink_t < -1e30) std::cout << sink_t;

    BenchResult r;
    r.n = n;
    r.edges = poset.num_relations();
    r.hyperplanes = order_poly.num_of_hyperplanes();
    r.us_per_call_op = us_op;
    r.us_per_call_hp = us_hp;
    r.speedup = us_hp / us_op;
    return r;
}


int main() {
    std::cout << "================================================================" << std::endl;
    std::cout << "  Boundary Oracle Timing: OrderPolytope (sparse) vs HPolytope  " << std::endl;
    std::cout << "================================================================" << std::endl;
    std::cout << std::endl;
    std::cout << "Each row: average over 5000 calls on random (position, velocity)" << std::endl;
    std::cout << "pairs from the same poset built as both OrderPolytope and HPolytope." << std::endl;
    std::cout << std::endl;

    std::cout << std::left
              << std::setw(8) << "n"
              << std::setw(8) << "|E|"
              << std::setw(8) << "m"
              << std::setw(16) << "Sparse (us)"
              << std::setw(16) << "Dense (us)"
              << std::setw(12) << "Speedup"
              << std::endl;
    std::cout << std::string(68, '-') << std::endl;

    // Test configurations: (n, edge_probability)
    // edge_prob ~ 3/n gives roughly degree-3 sparse posets
    std::vector<std::pair<unsigned, double>> configs = {
        {8,   0.3},    // small, moderate density
        {16,  0.2},    // small-medium
        {32,  0.1},    // medium, sparse
        {64,  0.05},   // large, sparse (deg ~ 3)
        {128, 0.025},  // large
        {256, 0.012},  // very large, sparse
    };

    for (auto [n, p] : configs) {
        BenchResult r = run_benchmark(n, p);

        std::cout << std::left
                  << std::setw(8) << r.n
                  << std::setw(8) << r.edges
                  << std::setw(8) << r.hyperplanes
                  << std::setw(16) << std::fixed << std::setprecision(2) << r.us_per_call_op
                  << std::setw(16) << std::fixed << std::setprecision(2) << r.us_per_call_hp
                  << std::setw(12) << std::fixed << std::setprecision(1) << r.speedup << "x"
                  << std::endl;
    }

    std::cout << std::endl;
    std::cout << "Sparse = OrderPolytope::trigonometric_positive_intersect O(n+|E|)" << std::endl;
    std::cout << "Dense  = HPolytope::trigonometric_positive_intersect     O(mn)"     << std::endl;
    std::cout << std::endl;
    std::cout << "The speedup grows with n because the dense oracle does O(mn)" << std::endl;
    std::cout << "mat-vec products while the sparse oracle reads O(1) coordinates" << std::endl;
    std::cout << "per constraint row." << std::endl;

    return 0;
}
