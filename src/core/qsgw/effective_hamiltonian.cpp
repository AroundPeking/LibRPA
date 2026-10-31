#include "effective_hamiltonian.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace librpa_int
{
namespace qsgw
{
namespace
{

void validate_map(const SpinKMatrixMap& values,
                  const SpinKMatrixMap* layout,
                  const std::string& label)
{
    if (values.empty())
    {
        throw std::invalid_argument(
            "QSGW " + label + " Hamiltonian map is empty");
    }
    if (layout != nullptr && values.size() != layout->size())
    {
        throw std::invalid_argument(
            "QSGW " + label + " Hamiltonian spin layout differs");
    }

    for (const auto& [spin, by_kpoint] : values)
    {
        if (by_kpoint.empty())
        {
            throw std::invalid_argument(
                "QSGW " + label + " Hamiltonian spin map is empty");
        }
        const auto layout_spin = layout == nullptr
                                     ? SpinKMatrixMap::const_iterator{}
                                     : layout->find(spin);
        if (layout != nullptr &&
            (layout_spin == layout->end() ||
             by_kpoint.size() != layout_spin->second.size()))
        {
            throw std::invalid_argument(
                "QSGW " + label + " Hamiltonian k-point layout differs");
        }

        for (const auto& [kpoint, matrix] : by_kpoint)
        {
            if (matrix.nr() <= 0 || matrix.nr() != matrix.nc())
            {
                throw std::invalid_argument(
                    "QSGW " + label +
                    " Hamiltonian matrix must be non-empty and square");
            }
            if (layout != nullptr)
            {
                const auto layout_kpoint =
                    layout_spin->second.find(kpoint);
                if (layout_kpoint == layout_spin->second.end() ||
                    layout_kpoint->second.nr() != matrix.nr() ||
                    layout_kpoint->second.nc() != matrix.nc())
                {
                    throw std::invalid_argument(
                        "QSGW " + label +
                        " Hamiltonian keys or dimensions differ");
                }
            }

            for (int row = 0; row < matrix.nr(); ++row)
            {
                for (int column = 0; column < matrix.nc(); ++column)
                {
                    const cplxdb value = matrix(row, column);
                    if (!std::isfinite(value.real()) ||
                        !std::isfinite(value.imag()))
                    {
                        throw std::invalid_argument(
                            "QSGW " + label +
                            " Hamiltonian contains non-finite data");
                    }
                }
            }
        }
    }
}

} // namespace

SpinKMatrixMap build_reference_hamiltonian(
    const MeanField& reference)
{
    if (!reference.initialized())
    {
        throw std::invalid_argument(
            "QSGW reference mean field is not initialized");
    }
    SpinKMatrixMap result;
    for (int spin = 0; spin < reference.get_n_spins(); ++spin)
    {
        for (int kpoint = 0; kpoint < reference.get_n_kpoints(); ++kpoint)
        {
            Matz hamiltonian(reference.get_n_bands(),
                             reference.get_n_bands(), MAJOR::ROW);
            for (int band = 0; band < reference.get_n_bands(); ++band)
            {
                const double eigenvalue =
                    reference.get_eigenvals()[spin](kpoint, band);
                if (!std::isfinite(eigenvalue))
                {
                    throw std::invalid_argument(
                        "QSGW reference eigenvalue is non-finite");
                }
                hamiltonian(band, band) = eigenvalue;
            }
            result[spin][kpoint] = std::move(hamiltonian);
        }
    }
    return result;
}

SpinKMatrixMap assemble_effective_hamiltonian(const SpinKMatrixMap& reference_hamiltonian,
                                              const SpinKMatrixMap& dft_vxc,
                                              const SpinKMatrixMap& exchange,
                                              const SpinKMatrixMap& correlation,
                                              const SpinKMatrixMap* delta_hartree)
{
    validate_map(reference_hamiltonian, nullptr, "reference");
    validate_map(dft_vxc, &reference_hamiltonian, "DFT Vxc");
    validate_map(exchange, &reference_hamiltonian, "exchange");
    validate_map(correlation, &reference_hamiltonian, "correlation");
    if (delta_hartree) validate_map(*delta_hartree, &reference_hamiltonian, "Delta Hartree");
    SpinKMatrixMap result;
    for (const auto& [spin, by_kpoint] : reference_hamiltonian)
    {
        for (const auto& [kpoint, reference] : by_kpoint)
        {
            const Matz& vxc = dft_vxc.at(spin).at(kpoint);
            const Matz& exx = exchange.at(spin).at(kpoint);
            const Matz& vc = correlation.at(spin).at(kpoint);
            Matz assembled(reference.nr(), reference.nc(),
                           reference.major());
            for (int row = 0; row < reference.nr(); ++row)
            {
                for (int column = 0; column < reference.nc(); ++column)
                {
                    assembled(row, column) =
                        reference(row, column) - vxc(row, column) +
                        exx(row, column) + vc(row, column);
                    if (delta_hartree)
                        assembled(row, column) += delta_hartree->at(spin).at(kpoint)(row, column);
                }
            }
            for (int row = 0; row < assembled.nr(); ++row)
            {
                // Legacy Scheme A diagonalizes with eigsh(UPLO='U'). Keep
                // that upper triangle authoritative while materializing the
                // same operator as an explicitly Hermitian matrix.
                assembled(row, row) =
                    cplxdb(assembled(row, row).real(), 0.0);
                for (int column = row + 1;
                     column < assembled.nc(); ++column)
                {
                    assembled(column, row) =
                        std::conj(assembled(row, column));
                }
            }
            result[spin][kpoint] = std::move(assembled);
        }
    }
    return result;
}

SpinKMatrixMap build_qp_static_subtraction(const SpinKMatrixMap& entering_hamiltonian,
                                           const SpinKMatrixMap& reference_hamiltonian,
                                           const SpinKMatrixMap& dft_vxc,
                                           const SpinKMatrixMap* delta_hartree)
{
    validate_map(reference_hamiltonian, nullptr, "reference");
    validate_map(entering_hamiltonian, &reference_hamiltonian, "entering");
    validate_map(dft_vxc, &reference_hamiltonian, "DFT Vxc");
    if (delta_hartree) validate_map(*delta_hartree, &reference_hamiltonian, "Delta Hartree");
    SpinKMatrixMap result;
    for (const auto& [spin, by_k] : reference_hamiltonian)
        for (const auto& [k, reference] : by_k)
        {
            Matz value(reference.nr(), reference.nc(), reference.major());
            for (int i = 0; i != value.nr(); ++i)
                for (int j = 0; j != value.nc(); ++j)
                {
                    value(i, j) = entering_hamiltonian.at(spin).at(k)(i, j) - reference(i, j) +
                                  dft_vxc.at(spin).at(k)(i, j);
                    if (delta_hartree) value(i, j) -= delta_hartree->at(spin).at(k)(i, j);
                }
            result[spin][k] = std::move(value);
        }
    return result;
}

} // namespace qsgw
} // namespace librpa_int
