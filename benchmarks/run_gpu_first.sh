#!/usr/bin/env bash
# The campaign in priority tiers: the GPU-bound results the paper depends on
# first, the slow external-library sweeps last. Same variables and the same
# resumable steps as run_campaign.sh (finished steps are skipped), so it can
# be interrupted and restarted with the same OUT at any point.
#
#   DATA=~/libsvm OUT=benchmarks/results/<stamp> bash benchmarks/run_gpu_first.sh
#   TIERS="1" ...        # only tier 1;  TIERS="1 2" for the first two
#
# Tier 1 (hours): E8/E8b memory envelope, E10a-d solver quality at fixed lambda,
#                 E1a/E1e exact suite with TorchKM and linear baselines, E5 covtype
#                 budget curve.
# Tier 2 (hours to a day): E2a Adult scaling (TorchKM), E3a imbalanced, E4a scale
#                 (Nystrom), E6/E6b KQR, E7 DWD, E9 Table 2 (TorchKM + ThunderSVM).
# Tier 3 (days):  E1b-E1d, E2b, E3b, E4b, E4c, E9b: cuML, ThunderSVM, scikit-learn,
#                 Falkon and linear-baseline grid sweeps.

set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export OUT="${OUT:-benchmarks/results/$(date -u +%Y%m%dT%H%M%SZ)}"
TIERS="${TIERS:-1 2 3}"

declare -A TIER
TIER[1]="E8 E8b E10a E10b E10c E10d E1a E1e E5"
TIER[2]="E2a E3a E4a E6 E6b E7 E9"
TIER[3]="E1b E1c E1d E2b E3b E4b E4c E9b"

for t in $TIERS; do
    echo "=================== tier $t: ${TIER[$t]} ($(date -u +%Y-%m-%dT%H:%M:%SZ))"
    ONLY="${TIER[$t]}" bash "$HERE/run_campaign.sh"
done
echo "all requested tiers finished: $OUT"
