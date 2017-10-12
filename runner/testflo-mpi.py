#!/usr/bin/env python
#
#
# Usage: `testflo-mpi.py <args>`
#

import sys

from mpi4py import MPI

import testflo


print "running testflo with:", sys.argv[1:]
rc = testflo.main.main(sys.argv[1:])

print "testflo completed with rc:", rc
comm = MPI.COMM_WORLD
comm.send(rc)
