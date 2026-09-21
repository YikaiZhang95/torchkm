#!/usr/bin/env bash
# Download every LIBSVM file the campaign uses into $DATA (login node; the
# compute nodes usually have no internet). Compressed files stay compressed:
# the loaders read .bz2 and .xz directly. Resumable; skips files already there.
#
#   DATA=$SCRATCH/libsvm bash benchmarks/hpc/download_data.sh
#   SKIP_MNIST8M=1 ...    # skip the 1.3 GB mnist8m.scale.xz (scale suite only)

set -uo pipefail
DATA="${DATA:-${SCRATCH:-$HOME}/libsvm}"
BASE="https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets"
mkdir -p "$DATA"
cd "$DATA"

fetch() {  # fetch <subdir> <file>
    local url="$BASE/$1/$2"
    local plain="${2%.bz2}"; plain="${plain%.xz}"
    # skip if the file is present in any form: as named, decompressed, or compressed
    for f in "$2" "$plain" "$plain.bz2" "$plain.xz"; do
        if [[ -s "$f" ]]; then echo "have $f"; return; fi
    done
    echo "get  $2"
    if command -v wget >/dev/null; then wget -q -c -O "$2.part" "$url" && mv "$2.part" "$2"
    else curl -sS -L -C - -o "$2.part" "$url" && mv "$2.part" "$2"; fi
    [[ -s "$2" ]] || { echo "!! failed: $url"; rm -f "$2" "$2.part"; }
}

# Adult scaling study, Web page data (imbalanced), exact-range and scale suites
for f in a1a a1a.t a3a a3a.t a5a a5a.t a7a a7a.t a8a a8a.t a9a a9a.t w7a w7a.t w8a w8a.t; do
    fetch binary "$f"
done
fetch binary ijcnn1.bz2
fetch binary ijcnn1.t.bz2
fetch binary covtype.libsvm.binary.scale.bz2
fetch binary gisette_scale.bz2
fetch binary gisette_scale.t.bz2
# MNIST pairs (multiclass files; digits are selected at load time)
fetch multiclass mnist.scale.bz2
fetch multiclass mnist.scale.t.bz2
[[ "${SKIP_MNIST8M:-0}" == "1" ]] || fetch multiclass mnist8m.scale.xz
# Regression sets for kernel quantile regression
for f in cadata abalone cpusmall space_ga; do fetch regression "$f"; done
fetch regression YearPredictionMSD.bz2
fetch regression YearPredictionMSD.t.bz2

echo; echo "files in $DATA:"; ls -l "$DATA" | awk '{print $5, $9}' | column -t
