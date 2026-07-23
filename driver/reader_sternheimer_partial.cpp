#include "reader_sternheimer_partial.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>

#include "../src/io/fs.h"

namespace driver
{
namespace
{

std::string strip_comment(const std::string &line)
{
    const auto hash = line.find('#');
    const auto bang = line.find('!');
    return line.substr(0, std::min(hash, bang));
}

std::string resolve_response_path(const std::string &manifest_path,
                                  const std::string &response_file)
{
    const auto unresolved =
        librpa_int::is_absolute_path(response_file)
            ? response_file
            : librpa_int::join_path(librpa_int::parent_path(manifest_path), response_file);
    librpa_int::require_readable_file(unresolved);
    return std::filesystem::weakly_canonical(unresolved).string();
}

}  // namespace

void validate_sternheimer_partial_task_contract(const std::string &qpoint_manifest,
                                                const std::string &partial_manifest)
{
    if (!partial_manifest.empty() && qpoint_manifest.empty())
    {
        throw std::runtime_error("fn_sternheimer_partial_manifest requires fn_sternheimer_qpoints");
    }
}

std::vector<SternheimerPartialResponse> read_sternheimer_partial_manifest(const std::string &path)
{
    librpa_int::require_readable_file(path);
    std::ifstream input(path);
    std::vector<SternheimerPartialResponse> records;
    std::set<std::tuple<int, int, int>> seen_keys;
    std::string line;
    int line_number = 0;
    while (std::getline(input, line))
    {
        ++line_number;
        std::istringstream row(strip_comment(line));
        row >> std::ws;
        if (row.eof())
        {
            continue;
        }

        SternheimerPartialResponse record;
        std::string response_file;
        if (!(row >> record.iq >> record.ik_full >> record.ifreq >> response_file))
        {
            throw std::runtime_error(path + ": line " + std::to_string(line_number) +
                                     " must contain iq ik_full ifreq response_file");
        }
        std::string extra;
        if (row >> extra)
        {
            throw std::runtime_error(path + ": line " + std::to_string(line_number) +
                                     " contains extra fields");
        }
        if (record.iq <= 0)
        {
            throw std::runtime_error(path + ": line " + std::to_string(line_number) +
                                     " requires a positive one-based iq");
        }
        if (record.ik_full < 0)
        {
            throw std::runtime_error(path + ": line " + std::to_string(line_number) +
                                     " requires a non-negative zero-based ik_full");
        }
        if (record.ifreq <= 0)
        {
            throw std::runtime_error(path + ": line " + std::to_string(line_number) +
                                     " requires a positive one-based ifreq");
        }

        const auto key = std::make_tuple(record.iq, record.ik_full, record.ifreq);
        if (!seen_keys.insert(key).second)
        {
            throw std::runtime_error(
                path + ": line " + std::to_string(line_number) +
                ": duplicate (iq, ik_full, ifreq)=(" + std::to_string(record.iq) + ", " +
                std::to_string(record.ik_full) + ", " + std::to_string(record.ifreq) + ")");
        }
        record.response_path = resolve_response_path(path, response_file);
        records.push_back(std::move(record));
    }

    if (records.empty())
    {
        throw std::runtime_error(path + ": Sternheimer partial manifest is empty");
    }
    return records;
}

SternheimerPartialResponseGroups read_sternheimer_partial_response_groups(
    const std::vector<SternheimerPartialResponse> &records)
{
    if (records.empty())
    {
        throw std::runtime_error("Cannot group an empty Sternheimer partial-response list");
    }

    SternheimerPartialResponseGroups groups;
    for (const auto &record : records)
    {
        auto response = read_sternheimer_chi0_v1_matrix_file(record.response_path);
        if (response.iq != record.iq)
        {
            throw std::runtime_error(record.response_path +
                                     ": binary iq=" + std::to_string(response.iq) +
                                     " does not match manifest iq=" + std::to_string(record.iq));
        }
        if (response.ifreq != record.ifreq)
        {
            throw std::runtime_error(
                record.response_path + ": binary ifreq=" + std::to_string(response.ifreq) +
                " does not match manifest ifreq=" + std::to_string(record.ifreq));
        }
        if (!std::isfinite(response.omega) || !std::isfinite(response.weight) ||
            response.weight <= 0.0)
        {
            throw std::runtime_error(record.response_path +
                                     ": invalid Sternheimer frequency metadata");
        }

        const auto key = std::make_pair(record.iq, record.ifreq);
        auto [group_iter, inserted] = groups.try_emplace(key);
        auto &group = group_iter->second;
        if (inserted)
        {
            group.iq = response.iq;
            group.ifreq = response.ifreq;
            group.omega = response.omega;
            group.weight = response.weight;
            group.atom_naux = response.atom_naux;
        }
        else
        {
            const auto close = [](const double lhs, const double rhs) {
                return std::abs(lhs - rhs) <= 1e-12 * std::max({1.0, std::abs(lhs), std::abs(rhs)});
            };
            if (!close(group.omega, response.omega))
            {
                throw std::runtime_error(record.response_path +
                                         ": inconsistent omega within (iq, ifreq) group");
            }
            if (!close(group.weight, response.weight))
            {
                throw std::runtime_error(
                    record.response_path +
                    ": inconsistent frequency weight within (iq, ifreq) group");
            }
            if (group.atom_naux != response.atom_naux)
            {
                throw std::runtime_error(record.response_path +
                                         ": inconsistent atom_naux within (iq, ifreq) group");
            }
        }

        if (!group.representatives.emplace(record.ik_full, std::move(response.matrix)).second)
        {
            throw std::runtime_error("Duplicate Sternheimer representative response ik_full=" +
                                     std::to_string(record.ik_full) +
                                     " for iq=" + std::to_string(record.iq) +
                                     ", ifreq=" + std::to_string(record.ifreq));
        }
    }
    return groups;
}

}  // namespace driver
