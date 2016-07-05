#!/usr/bin/env sh

set -e

CACHE_PATH=cache/mpii-cooking
TRAIN_PATCHES="${CACHE_PATH}/train-patches"
VAL_PATCHES="${CACHE_PATH}/val-patches"
OUT_DIR="resized-patches"

if [ $# -ne 1 ]; then
    echo "USAGE: $0 <output size>" 1>&2
    exit 1
fi

out_size="$1"; shift
dest_dir="${OUT_DIR}/${out_size}px"
mkdir -p "$dest_dir"

for in_file in "$TRAIN_PATCHES"/*.h5; do
    out_file="${dest_dir}/train-$(basename "$in_file")"
    echo "[t] Resizing training datum $in_file to $out_file"
    ./keras/resize_dataset.py --out-size $out_size $@ "$in_file" "$out_file"
done

for in_file in "$VAL_PATCHES"/*.h5; do
    out_file="${dest_dir}/val-$(basename "$in_file")"
    echo "[v] Resizing validation datum $in_file to $out_file"
    ./keras/resize_dataset.py --out-size $out_size $@ "$in_file" "$out_file"
done
