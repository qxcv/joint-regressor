#!/usr/bin/env bash

./plot_pck.py \
    --input 'Fragkiadaki et al.~\cite{fragkiadaki2015recurrent}' ../stats/h36m/flfm-rnns/pcks.csv \
    --input 'Ours' ../stats/h36m/biposelets/pcks.csv \
    --dims 4.8 2 \
    --xmax 0.35 \
    --colnames Wrists Elbows \
    --no-thresh-px \
    $@
