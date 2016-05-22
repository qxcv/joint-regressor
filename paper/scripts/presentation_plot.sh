#!/usr/bin/env bash

./plot_pck.py \
    --input "Cherian et al." ../stats/mpii/cmas/pck-mpii-cmas.csv \
    --input "$(printf 'Pfister et al.\n(SpatialNet)')" ../stats/mpii/pcz/pck-mpii-pcz.csv \
    --input "Biposelets" ../stats/mpii/mine/2016-05-17-pck-mpii-mine.csv \
    --poster --dims 10 4 \
    $@

# Extra args: try --save awesome.svg (or similar) to get nice vector plot
