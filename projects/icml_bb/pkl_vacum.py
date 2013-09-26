"""
Vacume clean jobs pkl files
"""

import fnmatch
import os
import sys
import ipdb

def clean(path, exemptions):
    matches = []
    for root, dirnames, filenames in os.walk(path):
        for filename in fnmatch.filter(filenames, '*.pkl'):
            if root.split('/')[-1] not in exemptions:
                os.remove(os.path.join(root, filename))


if __name__  == "__main__":

    path = sys.argv[1]
    if len(sys.argv) > 2:
        exemptions = sys.argv[2:]
    else:
        exemptions = []
    clean(path, exemptions)
