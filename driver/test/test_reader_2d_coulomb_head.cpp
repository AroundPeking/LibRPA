#include <chrono>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>

#include "../reader_2d_coulomb_head.h"

namespace
{

class TempDirectory
{
public:
    std::filesystem::path path;

    TempDirectory()
    {
        path = std::filesystem::temp_directory_path() /
               ("librpa_2d_head_" +
                std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        std::filesystem::create_directory(path);
    }

    ~TempDirectory() { std::filesystem::remove_all(path); }
};

void require(bool condition, const char *message)
{
    if (!condition) throw std::runtime_error(message);
}

void require_missing_metadata_rejected(const std::string &current, const std::string &legacy)
{
    try
    {
        resolve_strict_2d_coulomb_head_metadata_path(current, legacy);
    }
    catch (const std::runtime_error &error)
    {
        require(std::string(error.what()).find("Missing strict 2D Coulomb metadata") !=
                    std::string::npos,
                "missing metadata error did not explain the expected filenames");
        return;
    }
    throw std::runtime_error("missing strict 2D metadata was accepted");
}

}  // namespace

int main()
{
    TempDirectory temp;
    const auto current = temp.path / "librpa_2d_coulomb_head.dat";
    const auto legacy = temp.path / "librpa_2d_coulomb_head.txt";

    std::ofstream(legacy) << "# legacy reader-v1\nversion = 1\n"
                             "area_parallel_bohr2 = 57.98686608670706\n"
                             "multipole_norm_squared = 446.68844283515915\n";
    require(resolve_strict_2d_coulomb_head_metadata_path(current.string(), legacy.string()) ==
                legacy.string(),
            "legacy metadata filename was not selected when .dat was absent");
    const auto metadata = read_strict_2d_coulomb_head_metadata(legacy.string());
    require(metadata.version == 1 && metadata.inplane_area_bohr2 == 57.98686608670706 &&
                metadata.auxiliary_monopole_norm_squared == 446.68844283515915,
            "legacy metadata values were not parsed exactly");

    std::ofstream(current) << "version = 1\narea_parallel_bohr2 = 10\n"
                              "multipole_norm_squared = 20\n";
    require(resolve_strict_2d_coulomb_head_metadata_path(current.string(), legacy.string()) ==
                current.string(),
            "current .dat metadata did not take precedence");

    std::filesystem::remove(current);
    std::filesystem::remove(legacy);
    require_missing_metadata_rejected(current.string(), legacy.string());
}
