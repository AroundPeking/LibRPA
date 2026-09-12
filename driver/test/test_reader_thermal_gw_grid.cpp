#include <mpi.h>

#include <chrono>
#include <cmath>
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
#include "../reader_thermal_gw_grid.h"

namespace
{
void require(bool condition, const char *message)
{
    if (!condition) throw std::runtime_error(message);
}

void rejects(const std::function<void()> &operation, const std::string &fragment = "")
{
    try
    {
        operation();
    }
    catch (const std::exception &error)
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
               ("librpa_thermal_gw_grid_" + std::to_string(rank) + "_" +
                std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        std::filesystem::create_directory(path);
    }
    ~TempDirectory() { std::filesystem::remove_all(path); }
};

// Unequal dimensions and deliberately unsorted signed labels expose transposes/reordering.
const std::string valid_file =
    "LIBRPA_THERMAL_GW_V1\nsparse-ir 2.1.4\nHa Ha^-1\n"
    "B exp_minus inverse F exp_plus integral\n8 1 2 3 1e-10\n2 3 4\n"
    "0.5 7\n2 0 -2\n1 -1 -2 0\n"
    "1 2\n0.125 0\n3 -4\n5 -6\n0.125 0\n7 8\n"
    "9 10\n11 -12\n13 14\n15 -16\n17 18\n19 -20\n21 22\n23 -24\n";

void write_file(const std::filesystem::path &path, const std::string &text)
{
    std::ofstream output(path, std::ios::binary);
    output << text;
    require(static_cast<bool>(output), "failed writing test input");
}

void check_reader()
{
    TempDirectory temp;
    const auto path = temp.path / "grid.dat";
    write_file(path, valid_file);
    const auto grid = driver::read_external_thermal_gw_grid(path.string());
    require(grid.package == "sparse-ir" && grid.package_version == "2.1.4", "lost provenance");
    require(grid.beta_ha_inv == 8 && grid.g_wmax_ha == 1 && grid.w_wmax_ha == 2 &&
                grid.sigma_wmax_ha == 3 && grid.tolerance == 1e-10,
            "lost GW spectral metadata");
    require(grid.ntau == 2 && grid.nboson == 3 && grid.nfermion == 4 &&
                grid.times == std::vector<double>({0.5, 7}),
            "incorrect rectangular dimensions or times");
    require(grid.bosonic_indices == std::vector<int>({2, 0, -2}) &&
                grid.fermionic_indices == std::vector<int>({1, -1, -2, 0}),
            "signed arbitrary order changed");
    require(grid.b_real == std::vector<double>({1, 0.125, 3, 5, 0.125, 7}) &&
                grid.b_imag == std::vector<double>({2, 0, -4, -6, 0, 8}) &&
                grid.f_real == std::vector<double>({9, 11, 13, 15, 17, 19, 21, 23}) &&
                grid.f_imag == std::vector<double>({10, -12, 14, -16, 18, -20, 22, -24}),
            "complex row-major B/F layout changed");

    auto edit = [&](const std::string &from, const std::string &to)
    {
        auto text = valid_file;
        const auto position = text.find(from);
        require(position != std::string::npos, "invalid test edit");
        text.replace(position, from.size(), to);
        write_file(path, text);
    };
    auto reject_edit =
        [&](const std::string &from, const std::string &to, const std::string &fragment)
    {
        edit(from, to);
        rejects([&]() { driver::read_external_thermal_gw_grid(path.string()); }, fragment);
    };
    reject_edit("GW_V1", "GW_V2", "version");
    reject_edit("GW_V1", "TAU_V1", "version");
    reject_edit("Ha Ha^-1", "eV Ha^-1", "units");
    reject_edit("Ha Ha^-1", "Ha au", "units");
    reject_edit("B exp_minus", "F exp_minus", "convention");
    reject_edit("exp_minus", "exp_plus", "convention");
    reject_edit("inverse", "integral", "convention");
    reject_edit("F exp_plus", "B exp_plus", "convention");
    reject_edit("exp_plus", "exp_minus", "convention");
    reject_edit("integral", "average", "convention");
    for (const auto &metadata :
         {"0 1 2 3 1e-10", "-8 1 2 3 1e-10", "8 0 2 3 1e-10", "8 1 -2 3 1e-10", "8 1 2 2.9 1e-10",
          "8 1 2 3 0", "8 1 2 3 1", "nan 1 2 3 1e-10", "8 inf 2 3 1e-10", "8 1 2 nan 1e-10",
          "8 1 2 3 inf", "1e309 1 2 3 1e-10", "8 1e308 1e308 1e308 1e-10", "1e308 1 2 3 1e-10",
          "8 1x 2 3 1e-10"})
        reject_edit("8 1 2 3 1e-10", metadata, "metadata");
    // Decimal equality must tolerate roundoff in g_wmax + w_wmax.
    edit("8 1 2 3 1e-10", "8 0.1 0.2 0.3 1e-10");
    driver::read_external_thermal_gw_grid(path.string());
    for (const auto &dimensions :
         {"0 3 4", "2 0 4", "2 3 0", "-2 3 4", "2 -3 4", "2 3 -4", "2147483647 3 4",
          "2 2147483647 4", "2 3 2147483647", "2147483648 3 4", "2.0 3 4"})
        reject_edit("2 3 4\n", std::string(dimensions) + "\n", "dimensions");
    reject_edit("2 3 4\n", "1000000 3 4\n", "size");
    reject_edit("2 3 4\n", "2 1000000 4\n", "size");
    reject_edit("2 3 4\n", "2 3 1000000\n", "size");
    for (const auto &times : {"0 7", "0.5 8", "7 0.5", "0.5 0.5", "nan 7", "0.5 inf"})
        reject_edit("0.5 7\n", std::string(times) + "\n", "time");
    for (const auto &indices :
         {"2 2 -2", "2 1 -2", "2 0 -3", "2147483648 0 -2", "-2147483648 0 2", "2.0 0 -2"})
        reject_edit("2 0 -2\n", std::string(indices) + "\n", "bosonic");
    for (const auto &indices :
         {"1 -1 -2 1", "1 -1 -3 0", "1 -1 -2 2147483648", "1 -1 -2 -2147483649", "1 -1 -2 0.0"})
        reject_edit("1 -1 -2 0\n", std::string(indices) + "\n", "fermionic");
    // Pairing must promote before negation: INT_MIN and INT_MAX are fermionic partners.
    edit("1 -1 -2 0\n", "2147483647 -1 -2147483648 0\n");
    const auto limits = driver::read_external_thermal_gw_grid(path.string());
    require(limits.fermionic_indices[0] == 2147483647 &&
                limits.fermionic_indices[2] == (-2147483647 - 1),
            "extreme signed fermionic labels changed");
    reject_edit("3 -4\n", "nan -4\n", "bosonic operator");
    reject_edit("3 -4\n", "3 inf\n", "bosonic operator");
    reject_edit("23 -24\n", "nan -24\n", "fermionic operator");
    reject_edit("23 -24\n", "23 inf\n", "fermionic operator");
    reject_edit("23 -24\n", "23-24\n", "fermionic operator");
    reject_edit("23 -24\n", "23 -24x\n", "fermionic operator");
    reject_edit("0.125 0\n", "0.0625 0\n", "normalization");
    reject_edit("0.125 0\n", "0.125 1e-8\n", "normalization");
    reject_edit("sparse-ir", std::string("sparse-") + char(0xff), "ASCII");
    reject_edit("2.1.4", std::string("2.1.4") + char(0), "ASCII");
    write_file(path, valid_file + "extra\n");
    rejects([&]() { driver::read_external_thermal_gw_grid(path.string()); }, "trailing");
    write_file(path, valid_file + char(0xff));
    rejects([&]() { driver::read_external_thermal_gw_grid(path.string()); }, "ASCII");
    // Every token boundary is a possible truncation, including the last imaginary value.
    std::istringstream tokens(valid_file);
    std::string prefix, token;
    while (tokens >> token)
    {
        write_file(path, prefix);
        rejects([&]() { driver::read_external_thermal_gw_grid(path.string()); });
        prefix += token + "\n";
    }
    write_file(path, prefix + " \t\r\n");
    driver::read_external_thermal_gw_grid(path.string());
    write_file(path, valid_file.substr(0, valid_file.size() - 1));
    driver::read_external_thermal_gw_grid(path.string());
    rejects([&]() { driver::read_external_thermal_gw_grid((temp.path / "absent").string()); },
            "open");
}

void initialize_reference(librpa::Handler &handler)
{
    handler.set_scf_dimension(1, 1, 2, 2);
    const double energies[] = {-0.25, 0.75}, occupations[] = {1, 1};
    handler.set_wg_ekb_efermi(1, 1, 2, occupations, energies, 0.0);
    handler.set_fermi_dirac_reference(0.125, 0.0, 2.0, 1e-10);
}

void check_loader(const std::string &fixture)
{
    TempDirectory temp;
    const auto path = temp.path / "grid.dat";
    write_file(path, valid_file);
    librpa::Handler handler(MPI_COMM_WORLD);
    librpa::Options options;
    options.tfgrids_type = LIBRPA_TFGRID_FD_MATSUBARA;
    options.nfreq = 2;
    options.ntau = 3;
    auto load = [&]() { driver::load_external_thermal_gw_grid(path.string(), handler, options); };
    rejects(load);
    handler.set_scf_dimension(1, 1, 2, 2);
    handler.set_fermi_dirac_reference(0.125, 0.0, 2.0, 1e-10);
    rejects(load);
    {
        const double energies[] = {-0.25, 0.75}, occupations[] = {1, 1};
        handler.set_wg_ekb_efermi(1, 1, 2, occupations, energies, 0.0);
    }
    handler.clear_fermi_dirac_reference();
    rejects(load);
    handler.set_fermi_dirac_reference(0.125, 0.0, 2.0, 1e-10);

    const double times[] = {0.5, 3, 7}, real[] = {1, 2, 5, 1, -3, 2};
    const double imag[] = {0, 0, 0, 2, -1, -1};
    handler.set_external_thermal_time_grid(8, 2, 1e-10, 2, 3, times, real, imag);
    // Established driver-test pattern: observe response metadata, never mutate internals.
    const auto ds = librpa_int::api::get_dataset_instance(handler);
    const auto *response = ds->external_thermal_time_grid.get();
    load();
    require(options.nfreq == 2 && options.ntau == 3, "GW loader changed response options");
    require(ds->external_thermal_time_grid.get() == response,
            "GW loader replaced response metadata");
    const auto parsed = driver::read_external_thermal_gw_grid(path.string());
    auto check_stored = [&](const driver::ExternalThermalGWGrid &expected)
    {
        require(static_cast<bool>(ds->external_thermal_gw_grid),
                "loader did not store GW metadata");
        const auto &stored = *ds->external_thermal_gw_grid;
        const auto &transform = stored.transform;
        require(stored.g_wmax_ha == expected.g_wmax_ha && stored.w_wmax_ha == expected.w_wmax_ha &&
                    stored.sigma_wmax_ha == expected.sigma_wmax_ha &&
                    stored.tolerance == expected.tolerance &&
                    transform.get_beta_ha_inv() == expected.beta_ha_inv &&
                    transform.get_times() == expected.times &&
                    transform.get_bosonic_indices() == expected.bosonic_indices &&
                    transform.get_fermionic_indices() == expected.fermionic_indices,
                "loader changed GW metadata or signed node order");
        const auto b = transform.copy_bosonic_frequency_to_time();
        const auto f = transform.copy_fermionic_time_to_frequency();
        require(b.nr == expected.ntau && b.nc == expected.nboson && f.nr == expected.nfermion &&
                    f.nc == expected.ntau,
                "loader transposed rectangular GW operators");
        for (int k = 0; k < b.size; ++k)
            require(b.c[k].real() == expected.b_real[k] && b.c[k].imag() == expected.b_imag[k],
                    "loader changed complex B coefficients");
        for (int k = 0; k < f.size; ++k)
            require(f.c[k].real() == expected.f_real[k] && f.c[k].imag() == expected.f_imag[k],
                    "loader changed complex F coefficients");
    };
    check_stored(parsed);
    librpa_int::initialize_ds_tfgrids(*ds, options);
    require(ds->tfg.get_time_nodes() == std::vector<double>({0.5, 3, 7}),
            "GW loader changed the response grid");
    handler.clear_external_thermal_gw_grid();
    require(!ds->external_thermal_gw_grid && ds->external_thermal_time_grid.get() == response,
            "clearing GW did not preserve independent response metadata");
    load();
    options.nfreq = 7;
    options.ntau = 11;
    load();
    require(options.nfreq == 7 && options.ntau == 11, "GW dimensions were copied into options");
    options.tfgrids_type = LIBRPA_TFGRID_MINIMAX;
    rejects(load, "fd_matsubara");
    options.tfgrids_type = LIBRPA_TFGRID_FD_MATSUBARA;
    handler.set_fermi_dirac_reference(0.25, 0.0, 2.0, 1e-10);
    rejects(load);
    handler.set_fermi_dirac_reference(0.125, 0.0, 2.0, 1e-10);
    const double wider_energies[] = {-0.25, 1.5}, occupations[] = {1, 1};
    handler.set_wg_ekb_efermi(1, 1, 2, occupations, wider_energies, 0.0);
    rejects(load, "g_wmax");
    const double energies[] = {-0.25, 0.75};
    handler.set_wg_ekb_efermi(1, 1, 2, occupations, energies, 0.0);
    load();
    check_stored(parsed);

    const auto grid = driver::read_external_thermal_gw_grid(fixture);
    require(grid.ntau == 19 && grid.nboson == 17 && grid.nfermion == 20,
            "frozen sparse-ir fixture dimensions changed");
    require(grid.package == "sparse-ir" && grid.package_version == "2.1.4" &&
                grid.beta_ha_inv == 8 && grid.g_wmax_ha == 1 && grid.w_wmax_ha == 2 &&
                grid.sigma_wmax_ha == 3,
            "frozen sparse-ir fixture provenance changed");
    require(grid.bosonic_indices.front() == -33 && grid.bosonic_indices.back() == 33 &&
                grid.fermionic_indices.front() == -48 && grid.fermionic_indices.back() == 47,
            "frozen signed sparse labels changed");
    require(std::abs(grid.b_real.front() - 3.2761450566677417) < 1e-14 &&
                std::abs(grid.b_imag.front() - 5.3949277677577481) < 1e-14,
            "frozen complex coefficient changed");
    driver::load_external_thermal_gw_grid(fixture, handler, options);
    require(options.nfreq == 7 && options.ntau == 11 &&
                ds->external_thermal_time_grid.get() == response,
            "frozen GW fixture changed response metadata");
    check_stored(grid);
}

void check_input()
{
    TempDirectory temp;
    const auto path = temp.path / "librpa.in";
    require(driver::DriverParams().fn_thermal_gw_grid.empty(), "GW metadata enabled by default");
    auto parse = [&](const std::string &task, const std::string &grid)
    {
        driver::driver_params = driver::DriverParams();
        driver::opts = librpa::Options();
        write_file(path, "task = " + task +
                             "\nfn_thermal_gw_grid = gw.dat\n"
                             "tfgrids_type = " +
                             grid + "\nnfreq = 7\nntau = 11\n");
        parse_inputfile_to_params(path.string());
    };
    for (const auto &task : {"rpa", "g0w0", "g0w0_band"})
    {
        parse(task, "fd_matsubara");
        require(driver::driver_params.fn_thermal_gw_grid == "gw.dat", "filename was not parsed");
        require(
            driver::driver_params.format().find("fn_thermal_gw_grid = gw.dat") != std::string::npos,
            "filename was not echoed");
        require(driver::driver_params.fn_thermal_tau_grid.empty() && driver::opts.nfreq == 7 &&
                    driver::opts.ntau == 11,
                "GW option changed the response grid");
    }
    rejects([&]() { parse("exx", "fd_matsubara"); }, "task");
    rejects([&]() { parse("rpa", "minimax"); }, "fd_matsubara");
    rejects([&]() { parse("g0w0", "minimax"); }, "fd_matsubara");
    driver::driver_params = driver::DriverParams();
    driver::opts = librpa::Options();
    write_file(path, "task = exx\n");
    parse_inputfile_to_params(path.string());
    require(driver::driver_params.fn_thermal_gw_grid.empty(), "absent option enabled GW input");
}

void check_rank_local_failure()
{
    TempDirectory temp;
    int rank = 0, size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    require(size == 2, "rank-local failure test requires two ranks");
    librpa::Handler handler(MPI_COMM_WORLD);
    initialize_reference(handler);
    librpa::Options options;
    options.tfgrids_type = LIBRPA_TFGRID_FD_MATSUBARA;
    const auto path = temp.path / "grid.dat";
    for (int failure = 0; failure < 2; ++failure)
    {
        write_file(path, valid_file + (rank == 1 && failure == 0 ? "trailing\n" : ""));
        if (rank == 1 && failure == 1) handler.set_fermi_dirac_reference(0.25, 0.0, 2.0, 1e-10);
        int local_failed = 0, any_failed = 0, failed_count = 0;
        try
        {
            driver::load_external_thermal_gw_grid(path.string(), handler, options);
        }
        catch (const std::exception &)
        {
            local_failed = 1;
        }
        MPI_Allreduce(&local_failed, &any_failed, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        MPI_Allreduce(&local_failed, &failed_count, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        require(any_failed == 1 && failed_count == 1 && local_failed == rank,
                "rank-local loader failure was not coordinated");
    }
}
}  // namespace

int main(int argc, char **argv)
{
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    int failed = 0;
    try
    {
        if (argc == 2 && std::string(argv[1]) == "--rank-local-failure")
            check_rank_local_failure();
        else
        {
            require(argc == 2, "supply the frozen GW fixture path");
            check_reader();
            check_loader(argv[1]);
            check_input();
        }
    }
    catch (const std::exception &error)
    {
        std::cerr << error.what() << '\n';
        failed = 1;
    }
    int any_failed = 0;
    MPI_Allreduce(&failed, &any_failed, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    MPI_Finalize();
    return any_failed;
}
