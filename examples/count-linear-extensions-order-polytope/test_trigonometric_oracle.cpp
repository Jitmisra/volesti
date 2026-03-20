// volesti — trigonometric oracle tests for order polytopes
// Test: Validate OrderPolytope::trigonometric_positive_intersect
//       against HPolytope::trigonometric_positive_intersect
//
// For the same polytope (constructed as both HPolytope and OrderPolytope),
// the two boundary oracles must return identical (t_min, facet) results.
// This validates our O(n+|E|) implementation.

#include <iostream>
#include <cmath>
#include <cassert>
#include <vector>
#include <random>
#include <iomanip>

#include "Eigen/Eigen"
#include "cartesian_geom/cartesian_kernel.h"
#include "cartesian_geom/point.h"
#include "convex_bodies/hpolytope.h"
#include "convex_bodies/orderpolytope.h"
#include "misc/poset.h"
#include "random_walks/random_walks.hpp"

// Boost RNG
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


// Build an HPolytope equivalent to the given OrderPolytope
// by extracting its constraint matrix A and vector b.
HP build_hpolytope_from_order_polytope(OP& op) {
    MT A = op.get_dense_mat();
    VT b = op.get_vec();
    return HP(op.dimension(), A, b);
}


// Test 1: Compare trigonometric_positive_intersect results
void test_oracle_equivalence(unsigned n, Poset::RV& relations, unsigned num_trials = 1000) {
    Poset poset(n, relations);
    OP order_poly(poset);
    HP h_poly = build_hpolytope_from_order_polytope(order_poly);

    // Get an inner point from the order polytope
    VT inner = order_poly.inner_point();
    Point p_inner(inner);

    std::mt19937 gen(42);
    std::normal_distribution<NT> normal(0.0, 1.0);
    std::uniform_real_distribution<NT> unif(0.5, 5.0);

    unsigned passed = 0;
    unsigned failed = 0;

    for (unsigned trial = 0; trial < num_trials; ++trial) {
        // Random position near the inner point
        VT pos = inner;
        for (unsigned i = 0; i < n; ++i)
            pos(i) += normal(gen) * 0.01;
        // Clamp to ensure inside [0,1]
        for (unsigned i = 0; i < n; ++i)
            pos(i) = std::max(0.01, std::min(0.99, pos(i)));
        // Check if point is inside
        Point p(pos);
        if (order_poly.is_in(p) == 0) continue;

        // Random velocity
        VT vel(n);
        for (unsigned i = 0; i < n; ++i)
            vel(i) = normal(gen);
        Point v(vel);

        // Random omega
        NT omega = unif(gen);

        // Test both oracles
        int facet_prev_op = -1;
        int facet_prev_hp = -1;

        auto [t_op, f_op] = order_poly.trigonometric_positive_intersect(p, v, omega, facet_prev_op);
        auto [t_hp, f_hp] = h_poly.trigonometric_positive_intersect(p, v, omega, facet_prev_hp);

        NT rel_err = std::abs(t_op - t_hp) / std::max(NT(1e-15), std::abs(t_hp));

        if (rel_err < 1e-6 && f_op == f_hp) {
            passed++;
        } else {
            failed++;
            if (failed <= 5) {
                std::cout << "  MISMATCH trial " << trial
                          << ": t_op=" << t_op << " t_hp=" << t_hp
                          << " f_op=" << f_op << " f_hp=" << f_hp
                          << " rel_err=" << rel_err << std::endl;
            }
        }
    }

    std::cout << "[Oracle Equivalence n=" << n << "] "
              << passed << "/" << (passed + failed) << " passed"
              << (failed == 0 ? " ✓" : " ✗") << std::endl;
}


// Test 2: Energy conservation — verify the HMC trajectory conserves energy
void test_energy_conservation(unsigned n, Poset::RV& relations) {
    Poset poset(n, relations);
    OP order_poly(poset);

    VT inner = order_poly.inner_point();
    Point p(inner);

    std::mt19937 gen(123);
    std::normal_distribution<NT> normal(0.0, 1.0);

    NT omega = 2.0;
    NT a = omega * omega / 2.0;  // a = omega^2/2

    // Random velocity
    VT vel(n);
    for (unsigned i = 0; i < n; ++i)
        vel(i) = normal(gen);
    Point v(vel);

    // Compute initial Hamiltonian: H = a*||x||^2 + 0.5*||v||^2
    NT H0 = a * p.squared_length() + 0.5 * v.squared_length();

    // Simulate trajectory with reflections
    NT T_total = 3.0;
    NT T_remaining = T_total;
    int facet_prev = -1;
    unsigned reflections = 0;

    while (T_remaining > 1e-12) {
        auto [t_hit, facet] = order_poly.trigonometric_positive_intersect(p, v, omega, facet_prev);

        if (T_remaining <= t_hit) {
            // No more reflections — advance to end
            NT sinVal = std::sin(omega * T_remaining);
            NT cosVal = std::cos(omega * T_remaining);
            for (unsigned i = 0; i < n; ++i) {
                NT np = cosVal * p[i] + (sinVal / omega) * v[i];
                NT nv = -omega * sinVal * p[i] + cosVal * v[i];
                p.set_coord(i, np);
                v.set_coord(i, nv);
            }
            break;
        }

        // Advance to boundary
        NT sinVal = std::sin(omega * t_hit);
        NT cosVal = std::cos(omega * t_hit);
        for (unsigned i = 0; i < n; ++i) {
            NT np = cosVal * p[i] + (sinVal / omega) * v[i];
            NT nv = -omega * sinVal * p[i] + cosVal * v[i];
            p.set_coord(i, np);
            v.set_coord(i, nv);
        }

        // Reflect
        order_poly.compute_reflection(v, p, static_cast<unsigned int>(facet));
        T_remaining -= t_hit;
        reflections++;

        if (reflections > 10000) break;
    }

    NT Hf = a * p.squared_length() + 0.5 * v.squared_length();
    NT energy_err = std::abs(Hf - H0) / std::max(NT(1e-15), std::abs(H0));

    std::cout << "[Energy Conservation n=" << n << "] "
              << "H0=" << std::setprecision(10) << H0
              << " Hf=" << Hf
              << " rel_err=" << std::scientific << energy_err
              << " reflections=" << reflections
              << (energy_err < 1e-6 ? " ✓" : " ✗") << std::endl;
}


int main() {
    std::cout << "=== OrderPolytope Trigonometric Oracle Tests ===" << std::endl;
    std::cout << std::endl;

    // ---- Test on small chain poset: 0 < 1 < 2 < 3 ----
    {
        Poset::RV rels = {{0,1}, {1,2}, {2,3}};
        std::cout << "--- Chain poset n=4 (0<1<2<3) ---" << std::endl;
        test_oracle_equivalence(4, rels, 500);
        test_energy_conservation(4, rels);
    }

    // ---- Test on diamond poset: 0 < 1, 0 < 2, 1 < 3, 2 < 3 ----
    {
        Poset::RV rels = {{0,1}, {0,2}, {1,3}, {2,3}};
        std::cout << "\n--- Diamond poset n=4 ---" << std::endl;
        test_oracle_equivalence(4, rels, 500);
        test_energy_conservation(4, rels);
    }

    // ---- Test on antichain (no relations) ----
    {
        Poset::RV rels = {};
        std::cout << "\n--- Antichain n=5 (no relations) ---" << std::endl;
        test_oracle_equivalence(5, rels, 500);
        test_energy_conservation(5, rels);
    }

    // ---- Test on larger sparse poset n=8 ----
    {
        Poset::RV rels = {{0,2}, {1,3}, {2,4}, {3,5}, {4,6}, {5,7}};
        std::cout << "\n--- Sparse poset n=8, |E|=6 ---" << std::endl;
        test_oracle_equivalence(8, rels, 1000);
        test_energy_conservation(8, rels);
    }

    // ---- Test on dense poset n=6 ----
    {
        Poset::RV rels = {{0,1}, {0,2}, {0,3}, {1,4}, {2,4}, {2,5}, {3,5}, {4,5}};
        std::cout << "\n--- Dense poset n=6, |E|=8 ---" << std::endl;
        test_oracle_equivalence(6, rels, 1000);
        test_energy_conservation(6, rels);
    }

    // ---- Test on larger poset n=16 ----
    {
        Poset::RV rels = {{0,1}, {1,2}, {2,3}, {4,5}, {5,6}, {6,7},
                          {8,9}, {9,10}, {10,11}, {12,13}, {13,14}, {14,15},
                          {0,4}, {4,8}, {8,12}};
        std::cout << "\n--- Grid-like poset n=16, |E|=15 ---" << std::endl;
        test_oracle_equivalence(16, rels, 2000);
        test_energy_conservation(16, rels);
    }

    std::cout << "\n=== All Tests Complete ===" << std::endl;
    return 0;
}
