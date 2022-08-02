#!/usr/bin/python3
import logging
import numpy as np
from scipy.interpolate import LinearNDInterpolator
import random
import math
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator

def load_calibration():
    cal = {}
    cal['vs'] = np.loadtxt('vs.csv')
    cal['es'] = np.loadtxt('es.csv')
    # initialize interpolcation functions
    logging.info("Generate elevation interpolator")
    elevation = LinearNDInterpolator(cal['vs'][:,[0,3,4]], cal['es'])  # mx, ax, ay
    return elevation

def ref_check():
    cal = {}
    cal['vs'] = np.loadtxt('vs.csv')
    cal['es'] = np.loadtxt('es.csv')
    elevation = load_calibration()
    for sample in cal['vs']:
        e = elevation(sample[0], sample[3], sample[4])
        print(f"mx: {sample[0]}, ax: {sample[3]}, ay: {sample[4]}, e: {e}") 

def brute_force():
    elevation = load_calibration()
    while True:
        mx = random.uniform(-25,0)
        my = random.uniform(-50,40)
        mz = random.uniform(-120,-70)
        ax = random.uniform(-3,0)
        ay = random.uniform(-10,3)
        az = random.uniform(0,10)
        e = elevation(mx,my,mz,ax,ay,az)
        if not math.isnan(e):
            print(f"mx: {mx}, my: {my}, mz: {mz}, ax: {ax}, ay: {ay}, az: {az}, e: {e}") 

def plot():
    cal = {}
    cal['vs'] = np.loadtxt('vs.csv')
    cal['es'] = np.loadtxt('es.csv')
    elevation = load_calibration()
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    '''
    # Make data.
    X = cal['vs'][:,3]
    Y = cal['vs'][:,4]
    #Z = cal['es']
    Z = elevation(X,Y)

    # Plot the surface.
    surf = ax.scatter(X, Y, Z)

    '''
    # Make data.
    Xm = np.arange(-4, 1, 0.25)
    Ym = np.arange(-3, 5, 0.25)
    Xm, Ym = np.meshgrid(Xm, Ym)
    Zm = elevation(Xm, Ym)

    # Plot the surface.
    surf = ax.plot_surface(Xm, Ym, Zm, cmap=cm.coolwarm,
                                   linewidth=0, antialiased=False)
    # Customize the z axis.
    ax.set_zlim(-1.51, 1.51)
    ax.zaxis.set_major_locator(LinearLocator(10))
    # A StrMethodFormatter is used automatically
    ax.zaxis.set_major_formatter('{x:.02f}')

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    fig.savefig('plot.pdf')

