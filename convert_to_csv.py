#!/usr/bin/python3
import numpy as np
vs = np.loadtxt('vs.csv')
np.savetxt('vs_csv.csv', vs, delimiter=',', fmt='%f', header='mx,my,mz,ax,ay,az', comments='')
vs = np.loadtxt('es.csv')
np.savetxt('es_csv.csv', vs, delimiter=',', fmt='%f', header='es', comments='')
vs = np.loadtxt('hs.csv')
np.savetxt('hs_csv.csv', vs, delimiter=',', fmt='%f', header='hs', comments='')
