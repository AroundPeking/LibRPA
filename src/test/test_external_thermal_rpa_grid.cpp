#include <cmath>
#include <limits>
#include <stdexcept>

#include "../core/timefreq.h"

using namespace librpa_int;

int main()
{
    const double beta = 8.0, two_pi = 2.0 * std::acos(-1.0);
    const std::vector<double> times{1.0, 3.0, 5.0, 7.0};
    const std::vector<int> indices{0, 3, 41};
    const std::vector<double> weights{0.5 / beta, -0.25, 2.0};
    ComplexMatrix transform(3, 4);
    for (int j = 0; j < 4; ++j) transform(0, j) = 2.0;
    transform(1, 0) = {1.0, 0.3};
    transform(1, 1) = {-1.0, -0.3};
    TFGrids grid(3);
    grid.set_finite_beta_rpa_grid(times, beta, indices, weights, transform);
    auto check = [&]()
    {
        if (!grid.has_sparse_finite_beta_rpa_sum()) throw std::runtime_error("missing sparse flag");
        for (int i = 0; i < 3; ++i)
            if (std::abs(grid.get_freq_nodes()[i] - two_pi * indices[i] / beta) > 1e-13 ||
                std::abs(grid.find_correlation_frequency_weight(grid.get_freq_nodes()[i]) -
                         weights[i]) > 1e-13)
                throw std::runtime_error("explicit Matsubara indices or RPA weights lost");
    };
    check();
    auto reject = [&](std::vector<int> modes, std::vector<double> sum_weights)
    {
        bool rejected = false;
        try
        {
            grid.set_finite_beta_rpa_grid(times, beta, modes, sum_weights, transform);
        }
        catch (const std::runtime_error &)
        {
            rejected = true;
        }
        if (!rejected) throw std::runtime_error("invalid sparse RPA grid accepted");
        check();
    };
    reject({1, 3, 41}, weights);
    reject({0, 3, 3}, weights);
    reject({0, -1, 41}, weights);
    reject({0, 3}, weights);
    reject(indices, {0.5 / beta, 1.0});
    reject(indices, {1.0 / beta, 1.0, 1.0});
    reject(indices, {0.5 / beta, std::numeric_limits<double>::infinity(), 1.0});
    bool rejected = false;
    try
    {
        grid.finite_beta_power4_tail_weight(3);
    }
    catch (const std::runtime_error &)
    {
        rejected = true;
    }
    if (!rejected) throw std::runtime_error("sparse grid permits double-counting its tail");
    grid.set_finite_beta_time_grid(times, beta, transform);
    if (grid.has_sparse_finite_beta_rpa_sum()) throw std::runtime_error("stale sparse flag");
    grid.set_finite_beta_rpa_grid(times, beta, indices, weights, transform);
    grid.generate_finite_beta_matsubara(8, beta);
    if (grid.has_sparse_finite_beta_rpa_sum()) throw std::runtime_error("uniform reset failed");
    grid.set_finite_beta_rpa_grid(times, beta, indices, weights, transform);
    grid.reset(3);
    if (grid.has_sparse_finite_beta_rpa_sum()) throw std::runtime_error("grid reset failed");
}
