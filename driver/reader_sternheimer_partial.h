#pragma once

#include <map>
#include <string>
#include <utility>
#include <vector>

#include "reader_sternheimer.h"

namespace driver
{

struct SternheimerPartialResponse
{
    int iq = 0;
    int ik_full = -1;
    int ifreq = 0;
    std::string response_path;
};

struct SternheimerPartialResponseGroup
{
    int iq = 0;
    int ifreq = 0;
    double omega = 0.0;
    double weight = 0.0;
    std::vector<int> atom_naux;
    std::map<int, librpa_int::ComplexMatrix> representatives;
};

using SternheimerPartialResponseGroups =
    std::map<std::pair<int, int>, SternheimerPartialResponseGroup>;

void validate_sternheimer_partial_task_contract(const std::string &qpoint_manifest,
                                                const std::string &partial_manifest);

std::vector<SternheimerPartialResponse> read_sternheimer_partial_manifest(const std::string &path);

SternheimerPartialResponseGroups read_sternheimer_partial_response_groups(
    const std::vector<SternheimerPartialResponse> &records);

}  // namespace driver
