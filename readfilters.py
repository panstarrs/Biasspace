from matplotlib import rc
rc('font',**{'family':'sans-serif'})
rc('text', usetex=False)

from Tkinter import *

import numpy as np
import re
from collections import deque, OrderedDict
import os.path, sys


# Regular expressions for identifying parts of the filter file
num_regexp  = r"^# Number of filters in file = +([0-9]+)"
name_regexp = r"^# +([0-9]+) +([0-9]+) +(.*)"
meta_regexp = r"^# +([A-Z ]+): *(.*)"
head_regexp = r"^# +lambda +\[A\] +filter +response"


def read_filters(fname):

    # Read the filter file
    f = open(fname, "r")
    lines = deque(f.readlines())
    f.close()

    # Extract filter info
    filters = []
    num_filters = None
    this_filter = None
    while len(lines) > 0:
        line = lines.popleft()
        if line[0] == "#":
            # Check for number of filters
            m = re.match(num_regexp, line)
            if m is not None:
                num_filters = int(m.group(1))
                continue
            # Check for filter header
            m = re.match(name_regexp, line)
            if m is not None:
                if m.group(3) is not None:
                    name = m.group(3).strip()
                else:
                    name = ""
                this_filter = {"ifilter"     : int(m.group(1)),
                               "nlam"        : int(m.group(2)),
                               "name"        : name,
                               "data"        : [],
                               "metadata"    : {}}
                filters.append(this_filter)
                continue
            # Check for metadata
            m = re.match(meta_regexp, line)
            if m is not None:
                name  = m.group(1)
                value = [m.group(2),]
                # Check for continuation lines in DETAILS etc
                #while len(lines) > 0:
                #    if (re.match(meta_regexp, lines[0]) is None and
                #        re.match(head_regexp, lines[0]) is None and
                #        lines[0][0] == "#"):
                #        value.append(lines.popleft().lstrip("#").strip())
                #    else:
                #        break
                #this_filter["metadata"][name] = "\n".join(value)
                continue
        elif len(line.strip()) > 0:
            # This is a data line
            fields = line.split()
            wavelength = float(fields[0])
            response   = float(fields[1])
            this_filter["data"].append((wavelength, response))
        else:
            # Line is empty or contains only whitespace
            this_filter = None

    # Convert lists with (wavelength,response) pairs to arrays
    for fil in filters:
        wavelength, response = zip(*fil["data"])
        fil["wavelength"] = np.asarray(wavelength, dtype=float)
        fil["response"]   = np.asarray(response,   dtype=float)
        del fil["data"]
    # Consistency checks
    if len(filters) != num_filters:
        raise Exception("Number of filters read does not match header!")
    for fil in filters:
        if fil["wavelength"].shape[0] != fil["nlam"]:
            raise Exception("Number of data entries does not match filter header for filter "+fil["name"])

    return filters

if __name__ == "__main__":

   if len(sys.argv) != 2:
      print "Usage: filter_browser.py <filter_file>"
      sys.exit(1)

   print len(sys.argv)
   print sys.argv[1:2]
   filters = read_filters(sys.argv[1:2][0])
   print filters[1]['name']
   print filters[1]['wavelength']
   print filters[1]['response']
