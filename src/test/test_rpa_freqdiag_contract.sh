#!/bin/bash

set -euo pipefail

root=$(cd "$(dirname "$0")/../.." && pwd)
source_file=$root/src/core/epsilon.cpp

test -s "$source_file"
grep -q 'global::should_output(LIBRPA_VERBOSE_DEBUG) && comm_h.is_root()' "$source_file"
grep -q 'RPA normal split ifreq=%d' "$source_file"
grep -q 'RPA freqdiag ifreq=%d freq=' "$source_file"
grep -q 'trace_pi_diag.real()' "$source_file"
grep -q 'ln_det_diag.real()' "$source_file"
grep -q 'rpa_for_omega_q.real()' "$source_file"

echo RPA_FREQDIAG_CONTRACT_OK
