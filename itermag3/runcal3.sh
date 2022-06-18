#!/bin/bash
echo ". gendat parameters:"
echo ". p = ellipsoid axis gains, xyz. A unit sphere is -p 1 1 1"
echo ".      -p 1.0 1.2 0.9"
echo ". c = center of ellipsoid in units of radius of unit sphere."
echo ". r = rotations to tilt the ellipsoid in degrees, roll, pitch, heading"
echo ".      -r 5.0 10.0 15.0"
echo ". a = asymmetry, in normalized units, e.g. 0.01 is one percent."
echo ".      -a 0.01 0.15 0.15"
echo ". g = gain. converts counts in unit sphere to milli Gauss, "
echo ".     e.g about 400 in California: -g 400"
echo ". n = noise. 0.01 is one percent of local field. : -n 0.003 "
echo ".     Precision compasses are nearer to 0.001"
echo ". "
echo ". Because of the noise parameter, the data will be different each run."
python gendat3.py -P 1.0 1.05 0.95 -c 0.0625 0.125 0.1675 -r 10 20 30 -a 0.015 0.015 0.010 -g 400 -n 0.003 > gdrun.dat
python itermag3.py -f gdrun.dat
