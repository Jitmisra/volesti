// volesti — counting linear extensions via hmc on order polytopes
// Count Linear Extensions via Volume Computation on Order Polytopes
// Using Exact HMC with O(n+|E|) Trigonometric Boundary Oracle
//
// This program demonstrates end-to-end counting of linear extensions:
//   1. Read a poset from file (or use built-in examples)
//   2. Construct OrderPolytope
//   3. Compute Vol(O(P)) using cooling_gaussians with HMC
//   4. Return e(P) = n! * Vol(O(P))
//
// author: agnik

#include <iostream>
#include <fstream>
#include <cmath>
#include <iomanip>
#include <chrono>
#include <string>
#include <vector>

#include "Eigen/Eigen"
#include "cartesian_geom/cartesian_kernel.h"
#include "cartesian_geom/point.h"
#include "convex_bodies/hpolytope.h"
#include "convex_bodies/orderpolytope.h"
#include "misc/poset.h"
#include "misc/misc.h"
#include "random_walks/random_walks.hpp"
#include "volume/volume_cooling_gaussians.hpp"
#include "volume/volume_cooling_balls.hpp"
#include "generators/boost_random_number_generator.hpp"

#include <boost/random.hpp>

typedef double NT;
typedef Cartesian<NT> Kernel;
typedef typename Kernel::Point Point;
typedef HPolytope<Point> HP;
typedef OrderPolytope<Point> OP;
typedef Eigen::Matrix<NT, Eigen::Dynamic, Eigen::Dynamic> MT;
typedef Eigen::Matrix<NT, Eigen::Dynamic, 1> VT;
typedef BoostRandomNumberGenerator<boost::mt19937, NT> RNG;


// ============================================================================
//  count_linear_extensions — the core function
// ============================================================================
// Computes ln(e(P)) = ln(Vol(O(P))) + ln(n!)
// to avoid overflow for large n.
//
// Template parameter WalkType allows switching between:
//   - GaussianHamiltonianMonteCarloExactWalk (uses our O(n+|E|) oracle on OP)
//   - GaussianCDHRWalk (for comparison)
//   - GaussianBallWalk (for comparison)
template <typename WalkType>
NT count_linear_extensions_log(Poset& poset, NT epsilon = 0.1, unsigned walk_length = 1) {
    unsigned n = poset.num_elem();

    // 1. Construct OrderPolytope
    OP order_polytope(poset);

    // 2. Compute volume using cooling gaussians
    //    The key insight: volume_cooling_gaussians calls
    //    P.trigonometric_positive_intersect() which dispatches to our
    //    O(n+|E|) implementation on OrderPolytope
    NT volume = volume_cooling_gaussians<WalkType, RNG>(order_polytope, epsilon, walk_length);

    // 3. Compute ln(e(P)) = ln(Vol) + sum(ln(k) for k=1..n) = ln(Vol) + ln(n!)
    NT log_vol = std::log(volume);
    NT log_nfact = 0.0;
    for (unsigned k = 1; k <= n; ++k) {
        log_nfact += std::log(static_cast<NT>(k));
    }

    return log_vol + log_nfact;
}


// Same but using HPolytope (for comparison)
template <typename WalkType>
NT count_linear_extensions_log_hpoly(Poset& poset, NT epsilon = 0.1, unsigned walk_length = 1) {
    unsigned n = poset.num_elem();
    OP order_polytope(poset);
    // Extract as HPolytope
    MT A = order_polytope.get_dense_mat();
    VT b = order_polytope.get_vec();
    HP h_poly(n, A, b);

    NT volume = volume_cooling_gaussians<WalkType, RNG>(h_poly, epsilon, walk_length);

    NT log_vol = std::log(volume);
    NT log_nfact = 0.0;
    for (unsigned k = 1; k <= n; ++k) {
        log_nfact += std::log(static_cast<NT>(k));
    }

    return log_vol + log_nfact;
}


// Exact count by DP over order ideals (only works for small n)
// For a chain poset 0<1<...<(n-1), there is exactly 1 linear extension
// For an antichain of n elements, there are n! linear extensions
NT exact_log_count_chain(unsigned n) {
    return 0.0;  // ln(1) = 0
}

NT exact_log_count_antichain(unsigned n) {
    NT log_nfact = 0.0;
    for (unsigned k = 1; k <= n; ++k)
        log_nfact += std::log(static_cast<NT>(k));
    return log_nfact;
}


// ============================================================================
//  Benchmark runner
// ============================================================================
struct BenchmarkResult {
    std::string name;
    unsigned n;
    unsigned edges;
    NT log_count;
    NT exact_log_count;
    NT relative_error;
    double runtime_ms;
};

template <typename WalkType>
BenchmarkResult run_benchmark(const std::string& name, Poset& poset,
                               NT exact_log, NT epsilon, unsigned walk_length,
                               bool use_order_polytope = true) {
    BenchmarkResult result;
    result.name = name;
    result.n = poset.num_elem();
    result.edges = poset.num_relations();
    result.exact_log_count = exact_log;

    auto start = std::chrono::high_resolution_clock::now();

    if (use_order_polytope) {
        result.log_count = count_linear_extensions_log<WalkType>(poset, epsilon, walk_length);
    } else {
        result.log_count = count_linear_extensions_log_hpoly<WalkType>(poset, epsilon, walk_length);
    }

    auto end = std::chrono::high_resolution_clock::now();
    result.runtime_ms = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;

    if (exact_log > -1e30) {
        result.relative_error = std::abs(result.log_count - exact_log) / std::max(NT(1.0), std::abs(exact_log));
    } else {
        result.relative_error = -1.0;
    }

    return result;
}


void print_result(const BenchmarkResult& r) {
    std::cout << std::left << std::setw(25) << r.name
              << " n=" << std::setw(4) << r.n
              << " |E|=" << std::setw(4) << r.edges
              << " log_e=" << std::setw(12) << std::setprecision(4) << std::fixed << r.log_count;

    if (r.exact_log_count > -1e30) {
        std::cout << " exact=" << std::setw(12) << r.exact_log_count
                  << " rel_err=" << std::scientific << std::setprecision(2) << r.relative_error;
    }

    std::cout << " time=" << std::fixed << std::setprecision(1) << r.runtime_ms << "ms"
              << std::endl;
}


// ============================================================================
//  Main
// ============================================================================
int main(int argc, char* argv[]) {
    std::cout << "================================================================" << std::endl;
    std::cout << "  Counting Linear Extensions via Volume Computation" << std::endl;
    std::cout << "  using exact hmc on order polytopes" << std::endl;
    std::cout << "================================================================" << std::endl;
    std::cout << std::endl;

    NT epsilon = 0.1;
    unsigned walk_length = 1;

    // If a file is provided, use it
    if (argc >= 2) {
        std::string filename(argv[1]);
        std::ifstream data_file(filename);
        if (!data_file.is_open()) {
            std::cerr << "Error: Cannot open file " << filename << std::endl;
            return 1;
        }
        Poset poset = read_poset_from_file(data_file);
        unsigned n = poset.num_elem();

        std::cout << "Poset from file: " << filename << std::endl;
        std::cout << "  n = " << n << ", |E| = " << poset.num_relations() << std::endl;
        std::cout << std::endl;

        // Run with different walkers on OrderPolytope
        std::cout << "--- OrderPolytope (optimized) ---" << std::endl;

        std::cout << "[GaussianHMC]  ";
        auto r1 = run_benchmark<GaussianHamiltonianMonteCarloExactWalk>(
            "HMC-OrderPoly", poset, -1e31, epsilon, walk_length, true);
        print_result(r1);

        std::cout << "[GaussianCDHR] ";
        auto r2 = run_benchmark<GaussianCDHRWalk>(
            "CDHR-OrderPoly", poset, -1e31, epsilon, walk_length, true);
        print_result(r2);

        // Run with HPolytope for comparison
        std::cout << std::endl << "--- HPolytope (generic, for comparison) ---" << std::endl;

        std::cout << "[GaussianHMC]  ";
        auto r3 = run_benchmark<GaussianHamiltonianMonteCarloExactWalk>(
            "HMC-HPoly", poset, -1e31, epsilon, walk_length, false);
        print_result(r3);

        std::cout << "[GaussianCDHR] ";
        auto r4 = run_benchmark<GaussianCDHRWalk>(
            "CDHR-HPoly", poset, -1e31, epsilon, walk_length, false);
        print_result(r4);

        // Speedup
        if (r3.runtime_ms > 0) {
            std::cout << std::endl << "Speedup (HMC): "
                      << std::fixed << std::setprecision(1)
                      << (r3.runtime_ms / r1.runtime_ms) << "x" << std::endl;
        }

        return 0;
    }

    // ---- Built-in benchmarks ----
    std::cout << "Running built-in benchmarks (no input file provided)" << std::endl;
    std::cout << "Usage: " << argv[0] << " [poset_file.txt]" << std::endl;
    std::cout << std::endl;

    // ---- 1. Chain poset (exact count = 1) ----
    std::cout << "=== Chain Posets (exact count = 1, expected log_e ≈ 0) ===" << std::endl;
    for (unsigned n : {4, 6, 8}) {
        Poset::RV rels;
        for (unsigned i = 0; i < n - 1; ++i)
            rels.push_back({i, i + 1});
        Poset poset(n, rels);

        auto r = run_benchmark<GaussianHamiltonianMonteCarloExactWalk>(
            "Chain", poset, exact_log_count_chain(n), epsilon, walk_length, true);
        print_result(r);
    }

    // ---- 2. Antichain (exact count = n!) ----
    std::cout << std::endl << "=== Antichain Posets (exact count = n!) ===" << std::endl;
    for (unsigned n : {4, 5, 6}) {
        Poset::RV rels;
        Poset poset(n, rels);

        auto r = run_benchmark<GaussianHamiltonianMonteCarloExactWalk>(
            "Antichain", poset, exact_log_count_antichain(n), epsilon, walk_length, true);
        print_result(r);
    }

    // ---- 3. Diamond poset ----
    std::cout << std::endl << "=== Diamond Poset n=4 (exact count = 2, log_e ≈ 0.693) ===" << std::endl;
    {
        Poset::RV rels = {{0,1}, {0,2}, {1,3}, {2,3}};
        Poset poset(4, rels);
        auto r = run_benchmark<GaussianHamiltonianMonteCarloExactWalk>(
            "Diamond-4", poset, std::log(2.0), epsilon, walk_length, true);
        print_result(r);
    }

    // ---- 4. Comparison: OrderPolytope vs HPolytope ----
    std::cout << std::endl << "=== OrderPolytope vs HPolytope Comparison ===" << std::endl;
    for (unsigned n : {8, 16, 32}) {
        // Build sparse poset with avg degree ~2
        Poset::RV rels;
        for (unsigned i = 0; i < n - 1; i += 2)
            rels.push_back({i, i + 1});
        for (unsigned i = 0; i < n - 2; i += 3)
            rels.push_back({i, i + 2});
        Poset poset(n, rels);

        std::cout << "  n=" << n << " |E|=" << poset.num_relations() << ":" << std::endl;

        auto r_op = run_benchmark<GaussianHamiltonianMonteCarloExactWalk>(
            "HMC-OrderPoly", poset, -1e31, epsilon, walk_length, true);
        std::cout << "    ";
        print_result(r_op);

        auto r_hp = run_benchmark<GaussianHamiltonianMonteCarloExactWalk>(
            "HMC-HPoly", poset, -1e31, epsilon, walk_length, false);
        std::cout << "    ";
        print_result(r_hp);

        if (r_hp.runtime_ms > 0 && r_op.runtime_ms > 0) {
            std::cout << "    Speedup: " << std::fixed << std::setprecision(1)
                      << (r_hp.runtime_ms / r_op.runtime_ms) << "x" << std::endl;
        }
    }

    std::cout << std::endl << "=== Done ===" << std::endl;
    return 0;
}
