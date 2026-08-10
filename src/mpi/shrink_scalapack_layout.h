#pragma once

#include "base_blacs.h"

#include <stdexcept>

namespace librpa_int {

struct ShrinkScalapackLayout
{
    explicit ShrinkScalapackLayout(const BlacsCtxtHandler &blacs_h)
        : large_large(blacs_h), small_large(blacs_h), small_small(blacs_h)
    {
    }

    ArrayDesc large_large;
    ArrayDesc small_large;
    ArrayDesc small_small;
};

inline ShrinkScalapackLayout make_shrink_scalapack_layout(
    const BlacsCtxtHandler &blacs_h, const int n_large, const int n_small,
    const int block_size)
{
    if (n_large <= 0 || n_small <= 0 || block_size <= 0)
        throw std::invalid_argument("shrink ScaLAPACK dimensions and block size must be positive");

    ShrinkScalapackLayout layout(blacs_h);
    layout.large_large.init(n_large, n_large, block_size, block_size, 0, 0);
    layout.small_large.init(n_small, n_large, block_size, block_size, 0, 0);
    layout.small_small.init(n_small, n_small, block_size, block_size, 0, 0);
    return layout;
}

} // namespace librpa_int
