#!/usr/bin/env python2
"""Create a new HDF5 dataset from an old one, resizing (presumed) image data in
a supplied list of datasets."""

import argparse

import h5py
from scipy import ndimage

parser = argparse.ArgumentParser()
parser.add_argument('--field',
                    action='append',
                    dest='fields',
                    type=list,
                    default=['flow', 'images'])
parser.add_argument('--out-size', dest='out_size', type=int, default=32)
parser.add_argument('in_file', type=str)
parser.add_argument('out_file', type=str)

if __name__ == '__main__':
    args = parser.parse_args()
    tidy_name = lambda n: n.lstrip('/')  # flake8: noqa
    change_fields = {tidy_name(f) for f in args.fields}
    out_size = args.out_size
    with h5py.File(args.in_file, 'r') as in_fp:
        with h5py.File(args.out_file, 'w') as out_fp:

            def visitor(item_name):
                print('Visiting %s' % item_name)
                in_obj = in_fp[item_name]

                if not isinstance(in_obj, h5py.Dataset):
                    print('Skipping')
                    return

                if item_name in change_fields:
                    print('Resizing before copy')
                    if len(in_obj.shape) != 4:
                        raise Exception('Need 4D tensor to resize')

                    n, c, h, w = in_obj.shape
                    if h != w:
                        raise Exception(
                            'Dataset width not equal to height. Looks dodgy.')

                    if in_obj.chunks is not None:
                        batch_size = max([1, in_obj.chunks[0]])
                    else:
                        batch_size = n

                    out_obj = out_fp.create_dataset(
                        item_name,
                        shape=(n, c, out_size, out_size),
                        dtype=in_obj.dtype,
                        chunks=(batch_size, c, out_size, out_size))

                    zoom_factor = out_size / float(h)
                    zoom_arg = (1, 1, zoom_factor, zoom_factor)

                    # Copy across the resized dataset one element at a time
                    for start_idx in range(0, n, batch_size):
                        in_data = in_obj[start_idx:start_idx + batch_size]
                        resized = ndimage.zoom(in_data, zoom_arg)
                        out_obj[start_idx:start_idx + batch_size] = resized

                    return

                print('Copying unchanged')
                out_fp.copy(in_obj, item_name)

            in_fp.visit(visitor)
