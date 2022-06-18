
@echo off
echo . gendat parameters:
echo . p = ellipsoid axis gains, xyz. A unit sphere is -p 1 1 1
echo . c = center of ellipsoid in units of radius of unit sphere.
echo . r = rotations to tilt the ellipsoid in degrees, roll, pitch, heading
echo . a = asymmetry, in normalized units, e.g. 0.01 is one percent.
echo . g = gain. converts counts in unit sphere to milli Gauss, 
echo .     e.g about 400 in California
echo . n = noise. 0.01 is one percent of local field. 
echo .     Precision compasses are nearer to 0.001
echo .
echo . Because of the noise parameter, the data will be different each run.
echo on

::python gendat.py -P 1.0 1.3 0.9 -c 0.125 0.25 0.325 -r 10 20 30 -a 0.015 0.015 0.010 -g 400 -n 0.003 > gdrun.dat
python gendat3.py -P 1.0 1.05 0.95 -c 0.0625 0.125 0.1675 -r 10 20 30 -a 0.015 0.015 0.010 -g 400 -n 0.003 > gdrun.dat
python itermag3.py -f gdrun.dat