#!/usr/bin/env bash

./plot_pck.py \
    --input 'Cherian et al.~\cite{cherian2014mixing}' ../stats/mpii/cmas/pcks.csv \
    --input 'Chen \& Yuille~\cite{chen2014articulated}' ../stats/mpii/cy/pcks.csv \
    --input 'Combined~\cite{cherian2014mixing} and~\cite{chen2014articulated}' ../stats/mpii/comp2560/pcks.csv \
    --input 'Biposelets' ../stats/mpii/biposelets/pcks.csv \
    --dims 4.8 2 \
    --xmax 30 \
    $@
