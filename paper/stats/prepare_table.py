#!/usr/bin/env python3

"""Prepare a nice table of PCPs at different thresholds"""

from os import path

import pandas as pd

SOURCES = {
        'cmas': r'Cherian et al.~\cite{cherian2014mixing}',
        'cy': r'Chen \& Yuille~\cite{chen2014articulated}',
        'pcz': r'Pfister et al. SpatialNet~\cite{pfister2015flowing}',
        'comp2560': r'Combined~\cite{cherian2014mixing} and ~\cite{chen2014articulated}',
        'biposelets': r'Ours'
}
DATASETS = ['mpii', 'piw']
THRESHOLDS = [0.3, 0.5, 0.8]
LIMB_DICT = {
    'LowerArms': 'Lower arms',
    'UpperArms': 'Upper arms'
}
LIMBS, LIMB_NAMES = zip(*LIMB_DICT.items())

def get_row_data(ds, source):
    """For given dataset (``ds``) and source, get PCPs at each threshold for
    each limb."""
    csv_path = path.join(ds, source, 'pcps.csv')
    csv = pd.read_csv(csv_path)
    threshs = csv['Threshold']
    # Get lookup indices
    idxs = [(threshs == thresh).nonzero()[0][0] for thresh in THRESHOLDS]
    rv = []
    for limb in LIMBS:
        rv.extend(csv[limb][idxs])
    return rv

def to_pc(val):
    """Convert float value in [0, 1] to percentage"""
    return r'%.2f\%%' % (val * 100)

def prep_table(ds):
    """Prepare the entire table for a given dataset"""
    # Don't include a bar on the rightmost multicol (no bar on table right
    # side)
    bars = ['|'] * (len(LIMBS) - 1) + ['']
    limb_cols = [
        r'& \multicolumn{3}{c%s}{%s}' % (bar, name)
        for bar, name in zip(bars, LIMB_NAMES)
    ]
    limb_header = ' '.join(limb_cols) + r'\\'

    # Gives PCP thresholds, repeated for each limb
    thresh_headers = [' & '.join(map(str, THRESHOLDS))] * len(LIMBS)
    thresh_header = 'PCP threshold & ' + ' & '.join(thresh_headers) + r'\\'

    # Pretty rule
    nl = r'\tabucline-'

    # This is the actual data; one row per paper
    rv_lines = [limb_header, thresh_header, nl]
    for source, cite in SOURCES.items():
        row_data = get_row_data(ds, source)
        vals = ' & '.join(map(to_pc, row_data))
        line = '%s &\n%s\\\\' % (cite, vals)
        rv_lines.append(line)

    return '\n'.join(rv_lines)

if __name__ == '__main__':
    for ds in DATASETS:
        print('Table for ' + ds + ':')
        print('')
        print(prep_table(ds))
        print('\n' * 2)
