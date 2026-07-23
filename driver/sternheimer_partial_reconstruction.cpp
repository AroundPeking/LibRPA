#include "sternheimer_partial_reconstruction.h"

#include <algorithm>
#include <cmath>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>

#include "rpa_qsum.h"
#include "../src/core/symmetry_context.h"

namespace driver
{
namespace
{

std::vector<int> ordered_atom_naux(
    const std::map<librpa_int::atom_t, std::size_t> &atom_nabf)
{
    std::vector<int> result(atom_nabf.size(), -1);
    for (const auto &[atom, count] : atom_nabf)
    {
        const auto index = static_cast<std::size_t>(atom);
        if (index >= result.size() || count == 0)
        {
            throw std::runtime_error(
                "Sternheimer auxiliary basis has invalid atom indices or dimensions");
        }
        result[index] = static_cast<int>(count);
    }
    if (std::find(result.begin(), result.end(), -1) != result.end())
    {
        throw std::runtime_error("Sternheimer auxiliary basis atom indices are not contiguous");
    }
    return result;
}

librpa_int::Vector3_Order<double> q_vector(const SternheimerQPoint &point)
{
    return {point.q[0], point.q[1], point.q[2]};
}

}  // namespace

std::vector<SternheimerReconstructedResponse> reconstruct_sternheimer_partial_responses(
    const librpa_int::SymmetryContext &symmetry,
    const std::vector<librpa_int::SpeciesBasisLayout> &layouts,
    const std::map<librpa_int::atom_t, std::size_t> &atom_nabf,
    const std::vector<librpa_int::Vector3_Order<double>> &full_kpoints,
    const std::vector<SternheimerQPoint> &qpoints,
    const SternheimerPartialResponseGroups &groups,
    const int expected_nfreq,
    const bool use_rpa_gamma,
    const int lmax)
{
    if (expected_nfreq <= 0)
    {
        throw std::runtime_error("Sternheimer partial reconstruction requires positive nfreq");
    }
    if (qpoints.empty() || full_kpoints.empty())
    {
        throw std::runtime_error(
            "Sternheimer partial reconstruction requires non-empty q and full-k grids");
    }
    if (lmax < 0)
    {
        throw std::runtime_error("Sternheimer partial reconstruction requires ABF l-shell data");
    }

    const auto expected_atom_naux = ordered_atom_naux(atom_nabf);
    std::size_t represented_full_q_count = 0;
    for (const auto &point : qpoints)
    {
        const auto &qstar = librpa_int::find_symmetry_kstar_for_kpoint(
            symmetry.kstars, q_vector(point), "Sternheimer q-star coverage");
        represented_full_q_count += qstar.members.size();
    }
    if (represented_full_q_count != full_kpoints.size())
    {
        throw std::runtime_error(
            "Sternheimer q-star representatives do not cover the full q grid");
    }

    std::set<std::pair<int, int>> used_groups;
    std::vector<SternheimerReconstructedResponse> reconstructed;
    for (const auto &point : qpoints)
    {
        if (!use_rpa_gamma && is_rpa_gamma_point(point.q))
        {
            continue;
        }

        const auto q = q_vector(point);
        const auto little_group = librpa_int::build_sternheimer_fixed_q_little_group(
            symmetry.rspace_operations, q);
        const auto orbits = librpa_int::build_sternheimer_fixed_q_k_orbits(
            symmetry.rspace_operations, full_kpoints, q);
        for (int ifreq = 1; ifreq <= expected_nfreq; ++ifreq)
        {
            const auto key = std::make_pair(point.iq, ifreq);
            const auto group_iter = groups.find(key);
            if (group_iter == groups.end())
            {
                throw std::runtime_error(
                    "Sternheimer partial responses are missing (iq, ifreq)=(" +
                    std::to_string(point.iq) + ", " + std::to_string(ifreq) + ")");
            }
            used_groups.insert(key);
            const auto &group = group_iter->second;
            if (group.iq != point.iq || group.ifreq != ifreq)
            {
                throw std::runtime_error(
                    "Sternheimer partial response group key disagrees with its metadata");
            }
            if (group.atom_naux != expected_atom_naux)
            {
                throw std::runtime_error(
                    "Sternheimer partial response atom_naux does not match the active ABF basis");
            }
            if (!std::isfinite(group.omega) || !std::isfinite(group.weight)
                || group.weight <= 0.0)
            {
                throw std::runtime_error(
                    "Sternheimer partial response group has invalid frequency metadata");
            }

            auto matrix = librpa_int::reconstruct_sternheimer_fixed_q_response(
                symmetry, layouts, atom_nabf, q, orbits, group.representatives, lmax);
            auto qstar_responses = librpa_int::reconstruct_sternheimer_qstar_responses(
                symmetry, layouts, atom_nabf, q, matrix, lmax);
            const double q_weight =
                static_cast<double>(qstar_responses.size())
                / static_cast<double>(full_kpoints.size());
            const double q_weight_scale =
                std::max({1.0, std::abs(point.weight), std::abs(q_weight)});
            if (std::abs(point.weight - q_weight) > 1.0e-12 * q_weight_scale)
            {
                throw std::runtime_error(
                    "Sternheimer q-star weight disagrees with q-point manifest for iq="
                    + std::to_string(point.iq));
            }
            reconstructed.push_back({point.iq,
                                     ifreq,
                                     group.omega,
                                     group.weight,
                                     q_weight,
                                     static_cast<int>(full_kpoints.size()),
                                     static_cast<int>(orbits.size()),
                                     static_cast<int>(little_group.size()),
                                     std::move(matrix),
                                     std::move(qstar_responses)});
        }
    }

    if (used_groups.size() != groups.size())
    {
        throw std::runtime_error(
            "Sternheimer partial manifest contains unexpected q/frequency response groups");
    }
    return reconstructed;
}

SternheimerQStarRpaAudit compute_sternheimer_qstar_rpa_frequency(
    const SternheimerReconstructedResponse &response,
    const std::vector<librpa_int::SternheimerQStarResponse> &coulomb_qstar,
    const double sqrt_coulomb_threshold,
    const double invariance_tolerance)
{
    if (response.qstar_responses.empty()
        || response.qstar_responses.size() != coulomb_qstar.size())
    {
        throw std::runtime_error(
            "Sternheimer q-star response and Coulomb member counts disagree");
    }
    if (invariance_tolerance < 0.0)
    {
        throw std::runtime_error("Sternheimer q-star invariance tolerance must be non-negative");
    }

    SternheimerQStarRpaAudit audit;
    audit.qstar_size = static_cast<int>(response.qstar_responses.size());
    bool initialized = false;
    std::set<std::pair<int, int>> used_coulomb_members;
    for (const auto &response_member : response.qstar_responses)
    {
        const auto key = std::make_pair(response_member.star_index, response_member.member_index);
        const auto coulomb_iter = std::find_if(
            coulomb_qstar.begin(), coulomb_qstar.end(), [&key](const auto &member) {
                return std::make_pair(member.star_index, member.member_index) == key;
            });
        if (coulomb_iter == coulomb_qstar.end()
            || !used_coulomb_members.insert(key).second
            || !librpa_int::same_fractional_kpoint(
                response_member.q, coulomb_iter->q, 1.0e-8))
        {
            throw std::runtime_error(
                "Sternheimer q-star response and Coulomb members cannot be paired");
        }

        const auto member_result = librpa_int::compute_sternheimer_rpa_frequency(
            coulomb_iter->matrix,
            response_member.matrix,
            response.ifreq,
            response.omega,
            response.weight,
            response.q_weight,
            sqrt_coulomb_threshold);
        if (!initialized)
        {
            audit.result = member_result;
            initialized = true;
            continue;
        }

        const double difference = std::abs(member_result.integrand - audit.result.integrand);
        audit.max_integrand_difference = std::max(audit.max_integrand_difference, difference);
        const double scale =
            std::max({1.0, std::abs(member_result.integrand), std::abs(audit.result.integrand)});
        if (difference > invariance_tolerance * scale)
        {
            throw std::runtime_error("Sternheimer q-star trace-log invariance check failed");
        }
    }
    return audit;
}

}  // namespace driver
