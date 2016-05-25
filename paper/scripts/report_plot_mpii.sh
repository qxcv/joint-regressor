#!/usr/bin/env bash

./plot_pck.py \
    --input 'Cherian et al.~\cite{cherian2014mixing}' ../stats/mpii/cmas/pcks.csv \
    --input 'Chen \& Yuille~\cite{chen2014articulated}' ../stats/mpii/cy/pcks.csv \
    --input 'Pfister et al. (SpatialNet)~\cite{pfister2015flowing}' ../stats/mpii/pcz/pcks.csv \
    --dims 4.8 1.6 \
    --xmax 30 \
    $@
