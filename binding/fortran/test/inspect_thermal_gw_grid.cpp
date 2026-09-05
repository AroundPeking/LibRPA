#include "../../../src/api/instance_manager.h"

// Test-only inspection avoids adding a public API just to expose Dataset storage.
extern "C" int inspect_f03_thermal_gw_grid(int expected)
{
    const auto &instances = librpa_int::api::manager;
    // Instance zero is the manager's reserved null sentinel.
    if (instances.size() != 2 || instances[0] || !instances[1]) return 0;
    const auto &ds = *instances[1];
    if (!ds.mf.get_fermi_dirac_reference().enabled) return 0;
    if (!expected) return ds.external_thermal_gw_grid ? 0 : 1;
    if (!ds.external_thermal_gw_grid) return 0;
    const auto &g = *ds.external_thermal_gw_grid;
    const auto &t = g.transform;
    if (g.g_wmax_ha != 1 || g.w_wmax_ha != 2 || g.sigma_wmax_ha != 3 || g.tolerance != 1e-10 ||
        t.get_beta_ha_inv() != 8 || t.get_times() != std::vector<double>{0.5, 7} ||
        t.get_bosonic_indices() != std::vector<int>{2, 0, -2} ||
        t.get_fermionic_indices() != std::vector<int>{3, -1, 0, -4})
        return 0;
    const auto b = t.copy_bosonic_frequency_to_time();
    const auto f = t.copy_fermionic_time_to_frequency();
    if (b.nr != 2 || b.nc != 3 || f.nr != 4 || f.nc != 2) return 0;
    const double br[]{1, 0.125, 2, 3, 0.125, 4}, bi[]{2, 0, 3, 4, 0, 5};
    for (int i = 0; i < 6; ++i)
        if (b.c[i] != std::complex<double>(br[i], bi[i])) return 0;
    for (int i = 0; i < 8; ++i)
        if (f.c[i] != std::complex<double>(i + 1, 8 - i)) return 0;
    return 1;
}
