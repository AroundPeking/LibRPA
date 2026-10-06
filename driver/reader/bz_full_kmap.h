#pragma once

#include <iosfwd>
#include <vector>

namespace librpa::reader
{
struct ReaderContext;
bool read_full_kmap(std::istream &input, ReaderContext &ctx, const int nk[3],
                    const std::vector<double> &kvecs, const std::vector<double> &weights,
                    const std::vector<int> &qmap);
}  // namespace librpa::reader
