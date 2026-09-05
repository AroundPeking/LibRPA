#include <mpi.h>

#include <chrono>
#include <cmath>
#include <complex>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "../../src/api/dataset_helper.h"
#include "../../src/api/instance_manager.h"
#include "../driver.h"
#include "../inputfile.h"
#include "../reader_thermal_grid.h"

namespace
{
void require(bool value, const char *message)
{
    if (!value) throw std::runtime_error(message);
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

class TempDirectory
{
public:
    std::filesystem::path path;
    TempDirectory()
    {
        int rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        path = std::filesystem::temp_directory_path() /
               ("librpa_thermal_grid_" + std::to_string(rank) + "_" +
                std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        std::filesystem::create_directory(path);
    }
    ~TempDirectory() { std::filesystem::remove_all(path); }
};

const std::string valid_file =
    "LIBRPA_THERMAL_TAU_V1\nsparse-ir 2.1.4\nB exp_plus integral\n"
    "8 2 1e-10\n3 2\n0.5\n3\n7\n"
    "0\n1 0\n2 0\n5 0\n1\n1 2\n-3 -1\n2 -1\n";

void check_reader()
{
    TempDirectory temp;
    const auto path = temp.path / "grid.dat";
    auto write = [&](const std::string &text) { std::ofstream(path) << text; };
    write(valid_file);
    const auto grid = driver::read_external_thermal_grid(path.string());
    require(grid.package == "sparse-ir" && grid.package_version == "2.1.4", "lost provenance");
    require(grid.beta_ha_inv == 8.0 && grid.wmax_ha == 2.0 && grid.tolerance == 1e-10,
            "incorrect thermal metadata");
    require(grid.nfreq == 2 && grid.times == std::vector<double>({0.5, 3.0, 7.0}),
            "incorrect grid dimensions or nodes");
    require(grid.transform_real == std::vector<double>({1, 2, 5, 1, -3, 2}) &&
                grid.transform_imag == std::vector<double>({0, 0, 0, 2, -1, -1}),
            "complex operator layout changed");
    auto reject_edit =
        [&](const std::string &from, const std::string &to, const std::string &message)
    {
        auto text = valid_file;
        const auto position = text.find(from);
        require(position != std::string::npos, "invalid test edit");
        text.replace(position, from.size(), to);
        write(text);
        rejects([&]() { driver::read_external_thermal_grid(path.string()); }, message);
    };
    reject_edit("TAU_V1", "TAU_V2", "version");
    reject_edit("B exp_plus integral", "F exp_plus integral", "convention");
    reject_edit("exp_plus", "exp_minus", "convention");
    reject_edit("integral", "average", "convention");
    reject_edit("8 2 1e-10", "-8 2 1e-10", "metadata");
    reject_edit("3 2\n", "3 0\n", "dimensions");
    reject_edit("3 2\n", "2147483647 2147483647\n", "dimensions");
    reject_edit("3 2\n", "1000000 2\n", "size");
    reject_edit("0.5\n3\n7", "0.5\n0.5\n7", "time");
    reject_edit("0.5\n3\n7", "0\n3\n7", "time");
    reject_edit("0.5\n3\n7", "0.5\n3\n8", "time");
    reject_edit("5 0\n1\n", "5 0\n0\n", "index");
    reject_edit("-3 -1", "nan -1", "transform");
    write(valid_file.substr(0, valid_file.size() - 7));
    rejects([&]() { driver::read_external_thermal_grid(path.string()); }, "transform");
    write(valid_file + "extra\n");
    rejects([&]() { driver::read_external_thermal_grid(path.string()); }, "trailing");
    rejects([&]() { driver::read_external_thermal_grid((temp.path / "absent").string()); }, "open");

    write(valid_file);
    librpa::Handler handler(MPI_COMM_WORLD);
    handler.set_scf_dimension(1, 1, 2, 2);
    const double energies[] = {-0.25, 0.75}, occupations[] = {1.0, 1.0};
    handler.set_wg_ekb_efermi(1, 1, 2, occupations, energies, 0.0);
    handler.set_fermi_dirac_reference(0.125, 0.0, 2.0, 1e-10);
    librpa::Options options;
    options.nfreq = 2;
    options.ntau = 3;
    options.tfgrids_type = LIBRPA_TFGRID_FD_MATSUBARA;
    driver::load_external_thermal_grid(path.string(), handler, options);
    const auto ds = librpa_int::api::get_dataset_instance(handler);
    librpa_int::initialize_ds_tfgrids(*ds, options);
    require(ds->tfg.get_time_to_frequency_factor(1, 0) == std::complex<double>(1.0, 2.0),
            "loader did not reach public API");
    options.ntau = 4;
    rejects([&]() { driver::load_external_thermal_grid(path.string(), handler, options); },
            "dimensions");
    options.ntau = 3;
    options.tfgrids_type = LIBRPA_TFGRID_MINIMAX;
    rejects([&]() { driver::load_external_thermal_grid(path.string(), handler, options); },
            "fd_matsubara");
    options.tfgrids_type = LIBRPA_TFGRID_FD_MATSUBARA;
    handler.set_fermi_dirac_reference(0.25, 0.0, 2.0, 1e-10);
    rejects([&]() { driver::load_external_thermal_grid(path.string(), handler, options); }, "beta");

    const auto input = temp.path / "librpa.in";
    std::ofstream(input) << "task = rpa\nfn_thermal_tau_grid = grid.dat\n"
                            "tfgrids_type = fd_matsubara\nnfreq = 2\nntau = 3\n";
    parse_inputfile_to_params(input.string());
    require(driver::driver_params.fn_thermal_tau_grid == "grid.dat", "driver ignored filename");
    require(driver::DriverParams().fn_thermal_tau_grid.empty(), "external grid enabled by default");
}

void check_exported_file(const std::string &path)
{
    const auto grid = driver::read_external_thermal_grid(path);
    librpa::Handler handler(MPI_COMM_WORLD);
    handler.set_scf_dimension(1, 1, 2, 2);
    const double energies[] = {-0.25 * grid.wmax_ha, 0.25 * grid.wmax_ha};
    const double occupations[] = {2.0, 0.0};
    handler.set_wg_ekb_efermi(1, 1, 2, occupations, energies, 0.0);
    handler.set_fermi_dirac_reference(1.0 / grid.beta_ha_inv, 0.0, 2.0, 1e-8);
    librpa::Options options;
    options.nfreq = grid.nfreq;
    options.ntau = grid.times.size();
    options.tfgrids_type = LIBRPA_TFGRID_FD_MATSUBARA;
    driver::load_external_thermal_grid(path, handler, options);
    const auto ds = librpa_int::api::get_dataset_instance(handler);
    librpa_int::initialize_ds_tfgrids(*ds, options);
    require(ds->tfg.get_time_nodes() == grid.times, "exported nodes changed during loading");
    std::cout << "Exported operator accepted: " << grid.package << " " << grid.package_version
              << ", ntau=" << options.ntau << ", nfreq=" << options.nfreq << std::endl;
}
}  // namespace

int main(int argc, char **argv)
{
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    check_reader();
    if (argc == 2) check_exported_file(argv[1]);
    MPI_Finalize();
}
