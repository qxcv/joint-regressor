#!/usr/bin/env python3

# Hacky little downloader script to get H3.6M data from vision.imar.ro. H3.6M
# requires authentication to download, so you'll have to go to their website,
# fill out some form, agree to a EULA, wait for them to approve your account,
# login in to the website, and FINALLY download the data. If you want to use
# this script, then you'll need the PHPSESSID from an active login (on
# vision.imar.ro/human3.6m/). Going to that page and sticking
# "javascript:alert(document.cookie)" in the address bar should tell you what
# it is.

from argparse import ArgumentParser
from os.path import isfile, join
import re
from subprocess import call
from urllib.parse import urlparse, parse_qs

TO_FETCH = [
    'http://vision.imar.ro/human3.6m/filebrowser.php?download=1&filepath=Poses/D2_Positions&filename=SubjectSpecific_1.tgz&downloadname=S1',
    'http://vision.imar.ro/human3.6m/filebrowser.php?download=1&filepath=Poses/D2_Positions&filename=SubjectSpecific_6.tgz&downloadname=S5',
    'http://vision.imar.ro/human3.6m/filebrowser.php?download=1&filepath=Poses/D2_Positions&filename=SubjectSpecific_7.tgz&downloadname=S6',
    'http://vision.imar.ro/human3.6m/filebrowser.php?download=1&filepath=Poses/D2_Positions&filename=SubjectSpecific_2.tgz&downloadname=S7',
    'http://vision.imar.ro/human3.6m/filebrowser.php?download=1&filepath=Poses/D2_Positions&filename=SubjectSpecific_3.tgz&downloadname=S8',
    'http://vision.imar.ro/human3.6m/filebrowser.php?download=1&filepath=Poses/D2_Positions&filename=SubjectSpecific_4.tgz&downloadname=S9',
    'http://vision.imar.ro/human3.6m/filebrowser.php?download=1&filepath=Poses/D2_Positions&filename=SubjectSpecific_5.tgz&downloadname=S11',
    'http://vision.imar.ro/human3.6m/filebrowser.php?download=1&filepath=Videos&filename=SubjectSpecific_1.tgz&downloadname=S1',
    'http://vision.imar.ro/human3.6m/filebrowser.php?download=1&filepath=Videos&filename=SubjectSpecific_6.tgz&downloadname=S5',
    'http://vision.imar.ro/human3.6m/filebrowser.php?download=1&filepath=Videos&filename=SubjectSpecific_7.tgz&downloadname=S6',
    'http://vision.imar.ro/human3.6m/filebrowser.php?download=1&filepath=Videos&filename=SubjectSpecific_2.tgz&downloadname=S7',
    'http://vision.imar.ro/human3.6m/filebrowser.php?download=1&filepath=Videos&filename=SubjectSpecific_3.tgz&downloadname=S8',
    'http://vision.imar.ro/human3.6m/filebrowser.php?download=1&filepath=Videos&filename=SubjectSpecific_4.tgz&downloadname=S9',
    'http://vision.imar.ro/human3.6m/filebrowser.php?download=1&filepath=Videos&filename=SubjectSpecific_5.tgz&downloadname=S11',
    #'http://vision.imar.ro/human3.6m/filebrowser.php?download=1&filepath=MixedReality&filename=MixedReality.tgz',
]

def to_filename(url):
    return re.sub('[^\w\s-]', '', url).strip()

parser = ArgumentParser(description="Download necessary parts of H3.6M")
parser.add_argument(
    'sessid', type=str, help="PHPSESSID cookie from vision.imar.ro. You'll "
    "need to register with that website and then log in to obtain the cookie."
)
parser.add_argument(
    'destdir', type=str, help="Where to store downloaded files"
)

if __name__ == '__main__':
    args = parser.parse_args()

    # Grab sessid for vision.imar.ro (from browser cookie)
    sessid_string = 'PHPSESSID=' + args.sessid

    # Make destination directory
    dest_dir = args.destdir
    call(['mkdir', '-p', dest_dir])

    for url in TO_FETCH:
        # Destination filename
        parsed = urlparse(url)
        parsed_qs = parse_qs(parsed.query)
        prefix = to_filename(''.join(parsed_qs.get('filepath', [])))
        dest = prefix + ''.join(parsed_qs.get('filename', [to_filename(url)]))
        assert dest
        full_dest = join(dest_dir, dest)

        # Download
        if isfile(full_dest):
            print("'{}' exists so I won't download '{}'".format(full_dest, url))
        else:
            print("Downloading '{}' to '{}'".format(url, full_dest))
            call(['curl', '-b', sessid_string, '-o', full_dest, url])
