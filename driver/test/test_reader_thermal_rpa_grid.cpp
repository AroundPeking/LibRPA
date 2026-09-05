#include <mpi.h>

#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

#include "../../src/api/dataset_helper.h"
#include "../../src/api/instance_manager.h"
#include "../reader_thermal_grid.h"

namespace
{
void require(bool condition, const char *message)
{
    if (!condition) throw std::runtime_error(message);
}

void rejects(const std::function<void()> &operation, const std::string &fragment)
{
    try
    {
        operation();
    }
    catch (const std::runtime_error &error)
    {
        if (std::string(error.what()).find(fragment) != std::string::npos) return;
        throw;
    }
    throw std::runtime_error("expected error containing " + fragment);
}

struct TempDirectory
{
    std::filesystem::path path;
    TempDirectory()
    {
        int rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        path = std::filesystem::temp_directory_path() /
               ("librpa_thermal_rpa_grid_" + std::to_string(rank) + "_" +
                std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        std::filesystem::create_directory(path);
    }
    ~TempDirectory() { std::filesystem::remove_all(path); }
};

const std::string valid_file =
    "LIBRPA_THERMAL_RPA_V1\nsparse-ir 2.1.4\nB exp_plus integral\n"
    "8 2 4 1e-10\n3 4\n0.5\n3\n7\n"
    "0 0.0625\n1 0\n2 0\n5 0\n"
    "2 0.25\n1 2\n-3 -1\n2 -1\n"
    "9 -0.125\n0 0\n0 0\n0 0\n"
    "100 0\n0 0\n0 0\n0 0\n";

void check_reader()
{
    TempDirectory temp;
    const auto path = temp.path / "grid.dat";
    auto write = [&](const std::string &text) { std::ofstream(path) << text; };
    write(valid_file);
    const auto grid = driver::read_external_thermal_grid(path.string());
    require(grid.package == "sparse-ir" && grid.package_version == "2.1.4", "lost provenance");
    require(grid.beta_ha_inv == 8 && grid.wmax_ha == 2 && grid.rpa_wmax_ha == 4 &&
                grid.tolerance == 1e-10,
            "lost thermal metadata");
    require(grid.nfreq == 4 && grid.times == std::vector<double>({0.5, 3, 7}),
            "incorrect dimensions or times");
    require(grid.frequency_indices == std::vector<int>({0, 2, 9, 100}) &&
                grid.correlation_weights == std::vector<double>({0.0625, 0.25, -0.125, 0}),
            "lost sparse indices or signed weights");
    require(grid.transform_real == std::vector<double>({1, 2, 5, 1, -3, 2, 0, 0, 0, 0, 0, 0}) &&
                grid.transform_imag == std::vector<double>({0, 0, 0, 2, -1, -1, 0, 0, 0, 0, 0, 0}),
            "complex row-major layout changed");
    auto reject_edit = [&](const std::string &from, const std::string &to, const std::string &error)
    {
        auto text = valid_file;
        const auto position = text.find(from);
        require(position != std::string::npos, "invalid test edit");
        text.replace(position, from.size(), to);
        write(text);
        rejects([&]() { driver::read_external_thermal_grid(path.string()); }, error);
    };
    reject_edit("RPA_V1", "RPA_V2", "version");
    reject_edit("B exp_plus integral", "F exp_plus integral", "convention");
    reject_edit("exp_plus", "exp_minus", "convention");
    reject_edit("integral", "average", "convention");
    reject_edit("8 2 4 1e-10", "8 2 1.9 1e-10", "metadata");
    reject_edit("8 2 4 1e-10", "8 2 nan 1e-10", "metadata");
    reject_edit("8 2 4 1e-10", "-8 2 4 1e-10", "metadata");
    reject_edit("8 2 4 1e-10", "8 2 4 1", "metadata");
    reject_edit("3 4\n", "0 4\n", "dimensions");
    reject_edit("3 4\n", "2147483647 4\n", "dimensions");
    reject_edit("3 4\n", "1000000 4\n", "size");
    reject_edit("0.5\n3\n7", "0.5\n0.5\n7", "time");
    reject_edit("0.5\n3\n7", "0\n3\n7", "time");
    reject_edit("0.5\n3\n7", "0.5\n3\n8", "time");
    reject_edit("0 0.0625", "1 0.0625", "index");
    reject_edit("2 0.25", "-1 0.25", "index");
    reject_edit("9 -0.125", "2 -0.125", "index");
    reject_edit("9 -0.125", "1 -0.125", "index");
    reject_edit("0 0.0625", "0 0.0625000000125", "weight");
    reject_edit("0 0.0625", "0 -0.0625", "weight");
    reject_edit("2 0.25", "2 nan", "weight");
    reject_edit("2 0.25", "2 inf", "weight");
    reject_edit("-3 -1", "nan -1", "transform");
    write(valid_file.substr(0, valid_file.size() - 4));
    rejects([&]() { driver::read_external_thermal_grid(path.string()); }, "transform");
    write(valid_file + "extra\n");
    rejects([&]() { driver::read_external_thermal_grid(path.string()); }, "trailing");

    write(valid_file);
    librpa::Handler handler(MPI_COMM_WORLD);
    handler.set_scf_dimension(1, 1, 2, 2);
    const double energies[] = {-0.25, 0.75}, occupations[] = {1.0, 1.0};
    handler.set_wg_ekb_efermi(1, 1, 2, occupations, energies, 0.0);
    handler.set_fermi_dirac_reference(0.125, 0.0, 2.0, 1e-10);
    librpa::Options options;
    options.nfreq = 4;
    options.ntau = 3;
    options.tfgrids_type = LIBRPA_TFGRID_FD_MATSUBARA;
    auto load = [&]() { driver::load_external_thermal_grid(path.string(), handler, options); };
    load();
    const auto ds = librpa_int::api::get_dataset_instance(handler);
    librpa_int::initialize_ds_tfgrids(*ds, options);
    require(ds->external_thermal_time_grid->frequency_indices == grid.frequency_indices,
            "loader did not select RPA setter");
    require(std::abs(ds->tfg.find_correlation_frequency_weight(ds->tfg.get_freq_nodes()[2]) +
                     0.125) < 1e-14,
            "loader lost signed RPA sum weight");
    options.nfreq = 3;
    rejects(load, "dimensions");
    options.nfreq = 4;
    options.ntau = 4;
    rejects(load, "dimensions");
    options.ntau = 3;
    options.tfgrids_type = LIBRPA_TFGRID_MINIMAX;
    rejects(load, "fd_matsubara");
    options.tfgrids_type = LIBRPA_TFGRID_FD_MATSUBARA;
    auto unnormalized = valid_file;
    unnormalized.replace(unnormalized.find("1 0\n2 0\n5 0"), 3, "2 0");
    write(unnormalized);
    rejects(load, "normalization");
    require(ds->external_thermal_time_grid->frequency_indices == grid.frequency_indices,
            "rejected file changed stored grid");
    write(valid_file);
    handler.set_fermi_dirac_reference(0.25, 0.0, 2.0, 1e-10);
    rejects(load, "beta");
    rejects([&]() { librpa_int::initialize_ds_tfgrids(*ds, options); }, "beta");
    handler.set_fermi_dirac_reference(0.125, 0.0, 2.0, 1e-10);

    const std::string legacy_file =
        "LIBRPA_THERMAL_TAU_V1\nsparse-ir 2.1.4\nB exp_plus integral\n"
        "8 2 1e-10\n3 2\n0.5\n3\n7\n0\n1 0\n2 0\n5 0\n1\n1 2\n-3 -1\n2 -1\n";
    write(legacy_file);
    const auto legacy = driver::read_external_thermal_grid(path.string());
    require(legacy.frequency_indices.empty() && legacy.correlation_weights.empty(),
            "old format unexpectedly supplies sparse RPA weights");
    options.nfreq = 2;
    load();
    librpa_int::initialize_ds_tfgrids(*ds, options);
    require(ds->external_thermal_time_grid->frequency_indices.empty() &&
                std::abs(ds->tfg.find_correlation_frequency_weight(ds->tfg.get_freq_nodes()[1]) -
                         0.125) < 1e-14,
            "old loader path changed");
}
}  // namespace

int main(int argc, char **argv)
{
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    check_reader();
    MPI_Finalize();
}
