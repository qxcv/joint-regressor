#!/usr/bin/env bash

./plot_pck.py \
    --input 'Cherian et al.~\cite{cherian2014mixing}' ../stats/piw/cmas/pcks.csv \
    --input 'Chen \& Yuille~\cite{chen2014articulated}' ../stats/piw/cy/pcks.csv \
    --input 'Pfister et al. (SpatialNet)~\cite{pfister2015flowing}' ../stats/piw/pcz/pcks.csv \
    --input 'Combined~\cite{cherian2014mixing} and~\cite{chen2014articulated}' ../stats/piw/comp2560/pcks.csv \
    --input 'Ours' ../stats/piw/biposelets/pcks.csv \
    --dims 4.8 2 \
    --xmax 30 \
    $@
