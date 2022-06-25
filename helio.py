#!/usr/bin/python3

# Heliostat driver - this script should be run as a service

# In the end, this script should provide the remote IoT interface

# During development, it's invoked on command line

import adafruit_fxos8700
import adafruit_fxas21002c
from ahrs import Quaternion
from ahrs.filters import Madgwick, Mahony
import board
import copy
import csv
from gpiozero import Motor
import logging
import matplotlib
#matplotlib.use('gtkagg')
import matplotlib.pyplot as plt
import numpy as np
import random
from scipy import linalg, signal, interpolate
from simple_term_menu import TerminalMenu
import threading
import time
import vg

# Setup logging
logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d %(funcName)s] %(message)s', datefmt='%Y-%m-%d:%H:%M:%S', level=logging.INFO)

# Setup motors
MOTOR_TRAVEL_TIME = 10  # 30 seconds, 10 for speeding up durig debugging
MOTOR_S = Motor(26, 20)
MOTOR_T = Motor(19, 16)

# Setup sensor
i2c = board.I2C()  # uses board.SCL and board.SDA
sensor = adafruit_fxos8700.FXOS8700(i2c)

# Setup calibration data global variable
cal = { 'acxs': [], 'acys': [], 'aczs': [], 'tcxss': [], 'tcyss': [], 'tczss': [] }
NUMBER_OF_SCANS = 1


class Gyroscope(object):
    ''' Gyroscope class.
    '''

    def __init__(self):

        self.sensor = FXAS21002c()

    def read_rads(self):
        ''' Get a sample.

            Returns
            -------
            s : np.array
                The readings in radians/s (= raw values)
        '''
        return np.array(self.sensor.read())


class Accelerometer(object):
    ''' Accelerometer class.
    '''

    def __init__(self):

        self.sensor = FXOS8700_as()

    def read(self):
        ''' Get a sample.

            Returns
            -------
            s : np.array
                The reading as a unit vector [x,y,z]
        '''
        s = np.array(self.sensor.read())
        return s / np.linalg.norm(s)

    def read_ms2(self):
        ''' Get a sample.

            Returns
            -------
            s : np.array
                The readings in m/s^2 (= raw values)
        '''
        return np.array(self.sensor.read())


class Magnetometer(object):
    ''' Magnetometer class with calibration capabilities.
    '''

    def __init__(self):

        self.sensor = FXOS8700_ms()

        # initialize values
        self.b   = np.zeros([3, 1])
        self.A_1 = np.eye(3)
        self._calibration_s = np.array([])  # list of raw sensor readings during calibration
        self._calibration_complete = None

    def read_raw(self):
        ''' Get a sample.

            Returns
            -------
            s : np.array
                The reading in uT (not corrected).
        '''
        return np.array(self.sensor.read())

    def read_cal(self):
        ''' Get a sample.

            Returns
            -------
            s : np.array
                The reading in uT (corrected if performed calibration).
        '''
        s_vert = self.read_raw().reshape(3, 1)  # vertical array
        s_vert = np.dot(self.A_1, s_vert - self.b)
        s = s_vert[:,0]
        return s

    def save_calibration(self):
        ''' Saves calibration data to two files magcal_b.csv and magcal_A_1.csv
            in current directory
        '''

        logging.info("Magnetic calibration going to be saved to file")
        np.savetxt('magcal_b.csv', self.b)
        np.savetxt('magcal_A_1.csv', self.A_1)
        logging.info("Magnetic calibration saved to file")


    def load_calibration(self):
        ''' Loads calibration data from two files magcal_b.csv and magcal_A_1.csv
            in current directory
        '''

        logging.info("Magnetic calibration going to be loaded from file")
        self.b = np.loadtxt('magcal_b.csv')
        self.A_1 = np.loadtxt('magcal_A_1.csv')
        logging.info("Magnetic calibration loaded from file")


def launch_calibration_teslabs():
    ''' Performs calibration, actually only the loop collecting the data,
        running in a separate thread. It's monitoring the _calibration_complete
        property, set by the finish_calibration method, to stop. 
        The caller must take care of triggering the motion of the sensor and
        of calling finish_calibraiton when done.
    '''
    global magnetometer
    def collect_calibration_data():
        logging.info("Starting to collect samples for magnetometer calibration in thread")
        while magnetometer._calibration_complete == False:
            m = magnetometer.read_raw()  # sample in uT as np.array
            if np.shape(magnetometer._calibration_s)[0] == 0:
                magnetometer._calibration_s = m  # first sample
            else:
                magnetometer._calibration_s = np.vstack([magnetometer._calibration_s, m])
            print("x,y,z: {:8.1f},{:8.1f},{:8.1f}".format(m[0], m[1], m[2]), end="\r")
        logging.info("Stopped collecting samples for magnetometer calibration in thread")

    if magnetometer._calibration_complete == True:
        logging.error("Trying to launch calibration before previous one finished!")
        return
    magnetometer._calibration_s = np.array([])
    magnetometer._calibration_complete = False
    threading.Thread(target=collect_calibration_data).start()


def launch_calibration_precision():
    ''' Performs calibration, actually only the loop collecting the data,
        running in a separate thread. It's monitoring the _calibration_complete
        property, set by the finish_calibration method, to stop. 
        The caller must take care of triggering the motion of the sensor and
        of calling finish_calibraiton when done.
    '''
    global magnetometer, accelerometer
    def collect_calibration_data():
        logging.info("Starting to collect samples for magnetometer calibration in thread, including accelerometer data")
        while magnetometer._calibration_complete == False:
            m = magnetometer.read_raw()  # sample in uT as np.array
            a = accelerometer.read_ms2()  # sample as np.array
            if np.shape(magnetometer._calibration_s)[0] == 0:
                magnetometer._calibration_s = [m[0], m[1], m[2], a[0], a[1], a[2]]  # first sample
            else:
                magnetometer._calibration_s = np.vstack([magnetometer._calibration_s, [m[0], m[1], m[2], a[0], a[1], a[2]]])  # list of 6 numbers !
            print("magnetometer x,y,z: {:8.1f},{:8.1f},{:8.1f} - accelerometer x,y,z: {:8.1f},{:8.1f},{:8.1f}".format(m[0], m[1], m[2], a[0], a[1], a[2]), end="\r")
        logging.info("Stopped collecting samples for magnetometer calibration in thread")

    if magnetometer._calibration_complete == True:
        logging.error("Trying to launch calibration before previous one finished!")
        return
    magnetometer._calibration_s = np.array([])
    magnetometer._calibration_complete = False
    threading.Thread(target=collect_calibration_data).start()


def finish_calibration_teslabs():
    ''' Finishes the calibration, by setting the calibration_complete property
        to finish data collection by the launch_calibration thread and then
        calculating the calibration result
    '''
    global magnetometer
    def ellipsoid_fit(s):
        ''' Estimate ellipsoid parameters from a set of points.

            Parameters
            ----------
            s : array_like
              The samples (M,N) where M=3 (x,y,z) and N=number of samples.

            Returns
            -------
            M, n, d : array_like, array_like, float
              The ellipsoid parameters M, n, d.

            References
            ----------
            .. [1] Qingde Li; Griffiths, J.G., "Least squares ellipsoid specific
               fitting," in Geometric Modeling and Processing, 2004.
               Proceedings, vol., no., pp.335-340, 2004
        '''

        # D (samples)
        D = np.array([s[0]**2., s[1]**2., s[2]**2.,
                      2.*s[1]*s[2], 2.*s[0]*s[2], 2.*s[0]*s[1],
                      2.*s[0], 2.*s[1], 2.*s[2], np.ones_like(s[0])])

        # S, S_11, S_12, S_21, S_22 (eq. 11)
        S = np.dot(D, D.T)
        S_11 = S[:6,:6]
        S_12 = S[:6,6:]
        S_21 = S[6:,:6]
        S_22 = S[6:,6:]

        # C (Eq. 8, k=4)
        C = np.array([[-1,  1,  1,  0,  0,  0],
                      [ 1, -1,  1,  0,  0,  0],
                      [ 1,  1, -1,  0,  0,  0],
                      [ 0,  0,  0, -4,  0,  0],
                      [ 0,  0,  0,  0, -4,  0],
                      [ 0,  0,  0,  0,  0, -4]])

        # v_1 (eq. 15, solution)
        E = np.dot(linalg.inv(C),
                   S_11 - np.dot(S_12, np.dot(linalg.inv(S_22), S_21)))

        E_w, E_v = np.linalg.eig(E)

        v_1 = E_v[:, np.argmax(E_w)]
        if v_1[0] < 0: v_1 = -v_1

        # v_2 (eq. 13, solution)
        v_2 = np.dot(np.dot(-np.linalg.inv(S_22), S_21), v_1)

        # quadric-form parameters
        M = np.array([[v_1[0], v_1[3], v_1[4]],
                      [v_1[3], v_1[1], v_1[5]],
                      [v_1[4], v_1[5], v_1[2]]])
        n = np.array([[v_2[0]],
                      [v_2[1]],
                      [v_2[2]]])
        d = v_2[3]

        return M, n, d

    # finish reading sensor data
    magnetometer._calibration_complete = True
    s = magnetometer._calibration_s
    count = np.shape(magnetometer._calibration_s)[0]
    np.savetxt('magcal_data_raw_teslabs.csv', magnetometer._calibration_s)
    logging.info(f"Collected {count} samples for magnetometer calibration (check magcal_data_raw_teslabs.csv)")

    # average strength
    strength = 0
    for v in s:
        strength += np.linalg.norm(v) / count
    logging.info(f"Average strength of the samples is {strength} uT")

    # ellipsoid fit
    M, n, d = ellipsoid_fit(s.T)

    # calibration parameters
    # note: some implementations of sqrtm return complex type, taking real
    M_1 = linalg.inv(M)
    magnetometer.b = -np.dot(M_1, n)
    print(f"b {magnetometer.b}")
    magnetometer.A_1 = np.real(49.081 / np.sqrt(np.dot(n.T, np.dot(M_1, n)) - d) *
                       linalg.sqrtm(M))
    print(f"A_1 {magnetometer.A_1}")
    scal = np.dot(magnetometer.A_1, s.T - magnetometer.b).T
    np.savetxt('magcal_data_cal_teslabs.csv', scal)

    graphic_report(s, scal)

    # clear way for next calibration
    magnetometer._calibration_complete = None


def finish_calibration_precision():
    ''' Finishes the calibration, by setting the calibration_complete property
        to finish data collection by the launch_calibration thread and then
        calculating the calibration result
    '''
    global magnetometer
    def ellipsoid_iterate(mag,accel,verbose):

        def applyParams12(xyz,params):
            # first three of twelve parameters are the x,y,z offsets
            ofs=params[0:3]
            # next nine are the components of the 3x3 transformation matrix
            mat=np.reshape(params[3:12],(3,3))
            # subtract ofs
            xyzcentered=xyz-ofs
          
            xyzout=np.dot(mat,xyzcentered.T).T
            
            return xyzout

        def mgDot(mag,acc):
            ll=len(mag)
            mdg=np.zeros(ll)
            for ix in range(ll):
                mdg[ix]=np.dot(mag[ix],acc[ix].T)
            # print mdg   
            avgMdg=np.mean(mdg)
            stdMdg=np.std(mdg)
            # print avgMdg,stdMdg/avgMdg
            return (avgMdg,stdMdg)

        def magDotAccErr(mag,acc,mdg,params): #offset and transformation matrix from parameters 
            ofs=params[0:3]
            mat=np.reshape(params[3:12],(3,3))
            #subtract offset, then apply transformation matrix
            mc=mag-ofs
            mm=np.dot(mat,mc)
            #calculate dot product from corrected mags
            mdg1=np.dot(mm,acc)
            err=mdg-mdg1
            return err

        def errorEstimate(magN,accN,target,params):
            err2sum=0
            nsamp=len(magN)
            for ix in range(nsamp):
                err=magDotAccErr(magN[ix],accN[ix],target,params)
                err2sum += err*err
                # print "%10.6f" % (err)
            sigma=np.sqrt(err2sum/nsamp)  
            return sigma

        def analyticPartialRow(mag,acc,target,params):
            err0=magDotAccErr(mag,acc,target,params)
            # ll=len(params)
            slopeArr=np.zeros(12)
            slopeArr[0]=  -(params[3]*acc[0] + params[ 4]*acc[1] + params[ 5]*acc[2])
            slopeArr[1]=  -(params[6]*acc[0] + params[ 7]*acc[1] + params[ 8]*acc[2])
            slopeArr[2]=  -(params[9]*acc[0] + params[10]*acc[1] + params[11]*acc[2])
            
            slopeArr[ 3]= (mag[0]-params[0])*acc[0]
            slopeArr[ 4]= (mag[1]-params[1])*acc[0]
            slopeArr[ 5]= (mag[2]-params[2])*acc[0]
            
            slopeArr[ 6]= (mag[0]-params[0])*acc[1]
            slopeArr[ 7]= (mag[1]-params[1])*acc[1]
            slopeArr[ 8]= (mag[2]-params[2])*acc[1]
            
            slopeArr[ 9]= (mag[0]-params[0])*acc[2]
            slopeArr[10]= (mag[1]-params[1])*acc[2]
            slopeArr[11]= (mag[2]-params[2])*acc[2]
            
            return (err0,slopeArr)

        def normalize3(xyz):
            x=xyz[:,0]
            y=xyz[:,1]
            z=xyz[:,2]
            rarr = np.sqrt(x*x + y*y + z*z)
            ravg=np.mean(rarr)
            xyzn=xyz/ravg
            return (xyzn,ravg)

        def estimateCenter3D( arr, mode=0):

            # Slice off the component arrays
            xx=arr[:,0]
            yy=arr[:,1]
            zz=arr[:,2]
            
            #average point is centered sufficiently with well sampled data
            center=np.array([np.mean(xx),np.mean(yy),np.mean(zz)])
               
            #Center the samples
            xc=xx-center[0]
            yc=yy-center[1]
            zc=zz-center[2]
            
            # Calculate distance from center for each point 
            rc = np.sqrt(xc*xc + yc*yc + zc*zc)
            # Take the average
            radius = np.mean(rc)
            
            std = np.std(rc)
               
            return (center,radius,std)
   
        magCorrected=copy.deepcopy(mag)
        # Obtain an estimate of the center and radius
        # For an ellipse we estimate the radius to be the average distance
        # of all data points to the center
        (centerE,magR,magSTD)=estimateCenter3D(mag)
 
        #Work with normalized data
        magScaled=mag/magR
        centerScaled = centerE/magR
 
        (accNorm,accR)=normalize3(accel)
 
        params=np.zeros(12)
        #Use the estimated offsets, but our transformation matrix is unity
        params[0:3]=centerScaled
        mat=np.eye(3)
        params[3:12]=np.reshape(mat,(1,9))
 
        #initial dot based on centered mag, scaled with average radius
        magCorrected=applyParams12(magScaled,params)
        (avgDot,stdDot)=mgDot(magCorrected,accNorm)
 
        nSamples=len(magScaled)
        sigma = errorEstimate(magScaled,accNorm,avgDot,params)
        logging.info(f"Initial Sigma {sigma}")
 
        # pre allocate the data.  We do not actually need the entire
        # D matrix if we calculate DTD (a 12x12 matrix) within the sample loop
        # Also DTE (dimension 12) can be calculated on the fly.
 
        D=np.zeros([nSamples,12])
        E=np.zeros(nSamples)
 
        #If numeric derivatives are used, this step size works with normalized data.
        step=np.ones(12)
        step/=5000
 
        #Fixed number of iterations for testing.  In production you check for convergence
 
        nLoops=10
 
        for iloop in range(nLoops):
            # Numeric or analytic partials each give the same answer
            for ix in range(nSamples):
                #(f0,pdiff)=numericPartialRow(magScaled[ix],accNorm[ix],avgDot,params,step,1)
                (f0,pdiff)=analyticPartialRow(magScaled[ix],accNorm[ix],avgDot,params)
                E[ix]=f0
                D[ix]=pdiff
            #Use the pseudo-inverse
            DT=D.T
            DTD=np.dot(DT,D)
            DTE=np.dot(DT,E)
            invDTD=np.linalg.inv(DTD)
            deltas=np.dot(invDTD,DTE)

            p2=params + deltas

            sigma = errorEstimate(magScaled,accNorm,avgDot,p2)

            # add some checking here on the behavior of sigma from loop to loop
            # if satisfied, use the new set of parameters for the next iteration

            params=p2

            # recalculste gain (magR) and target dot product
            magCorrected=applyParams12(magScaled,params)
            (mc,mcR)=normalize3(magCorrected)
            (avgDot,stdDot)=mgDot(mc,accNorm)
            magR *= mcR
            magScaled=mag/magR
 
            logging.info("iloop {iloop}, sigma {sigma}")

        return (params,magR)

    def ellipsoid_iterate_symmetric(mag,verbose):
          
        def estimateCenter3D( arr, mode=0):

            # Slice off the component arrays
            xx=arr[:,0]
            yy=arr[:,1]
            zz=arr[:,2]
            
            #average point is centered sufficiently with well sampled data
            center=np.array([np.mean(xx),np.mean(yy),np.mean(zz)])
               
            #Center the samples
            xc=xx-center[0]
            yc=yy-center[1]
            zc=zz-center[2]
            
            # Calculate distance from center for each point 
            rc = np.sqrt(xc*xc + yc*yc + zc*zc)
            # Take the average
            radius = np.mean(rc)
            
            std = np.std(rc)
               
            return (center,radius,std)
       
        def ofsMatToParam9(ofs,mat,params):

            params[0:3]=ofs
           
            params[3]=mat[0,0]
            params[4]=mat[1,1]
            params[5]=mat[2,2]

            params[6]=mat[0,1]
            params[7]=mat[0,2]
            params[8]=mat[1,2]
          
            return params

        def radiusErr(mag,target,params):
            #offset and transformation matrix from parameters
            (ofs,mat)=param9toOfsMat(params)
           
            #subtract offset, then apply transformation matrix
            mc=mag-ofs
            mm=np.dot(mat,mc)

            radius = np.sqrt(mm[0]*mm[0] +mm[1]*mm[1] + mm[2]*mm[2] )
            err=target-radius
            return err

        def errorEstimateSymmetric(mag,target,params):
            err2sum=0
            nsamp=len(mag)
            for ix in range(nsamp):
                err=radiusErr(mag[ix],target,params)
                err2sum += err*err
                # print "%10.6f" % (err)
            sigma=np.sqrt(err2sum/nsamp)  
            return sigma  

        def errFn(mag,acc,target,params,mode):
            if mode == 1: return magDotAccErr(mag,acc,target,params)
            return radiusErr(mag,target,params)

        def numericPartialRow(mag,acc,target,params,step,mode):
           
            err0=errFn(mag,acc,target,params,mode)   
           
            ll=len(params)
            slopeArr=np.zeros(ll)
           
            for ix in range(ll):
           
                params[ix]=params[ix]+step[ix]
                errA=errFn(mag,acc,target,params,mode) 
                params[ix]=params[ix]-2.0*step[ix]
                errB=errFn(mag,acc,target,params,mode) 
                params[ix]=params[ix]+step[ix]
                slope= (errB-errA)/(2.0*step[ix])
                slopeArr[ix]=slope
              
            return (err0,slopeArr)

        def param9toOfsMat(params):
            ofs=params[0:3]
            mat=np.zeros(shape=(3,3))
           
            mat[0,0]=params[3]
            mat[1,1]=params[4]
            mat[2,2]=params[5]

            mat[0,1]=params[6]
            mat[0,2]=params[7]
            mat[1,2]=params[8]

            mat[1,0]=params[6]
            mat[2,0]=params[7]
            mat[2,1]=params[8]
            # print ofs,mat
            return (ofs,mat)

        (centerE,magR,magSTD)=estimateCenter3D(mag)
      
        magScaled=mag/magR
        centerScaled = centerE/magR
         
        params9=np.zeros(9)
        ofs=np.zeros(3)
        mat=np.eye(3)
        params9=ofsMatToParam9(centerScaled,mat,params9)
      
        nSamples=len(magScaled)
        sigma = errorEstimateSymmetric(magScaled,1,params9)
        if verbose: print ('Initial Sigma',sigma)
      
        step=np.ones(9)  
        step/=5000
        D=np.zeros([nSamples,9])
        E=np.zeros(nSamples)
        nLoops=10

        for iloop in range(nLoops):
     
            for ix in range(nSamples):
                (f0,pdiff)=numericPartialRow(magScaled[ix],magScaled[ix],1,params9,step,0)
                E[ix]=f0
                D[ix]=pdiff
            DT=D.T
            DTD=np.dot(DT,D)
            DTE=np.dot(DT,E)
            invDTD=np.linalg.inv(DTD)
            deltas=np.dot(invDTD,DTE)
     
            p2=params9 + deltas
           
            (ofs,mat)=param9toOfsMat(p2)
            sigma = errorEstimateSymmetric(magScaled,1,p2)
        
            params9=p2
         
            if verbose: 
                print ('iloop',iloop,'sigma',sigma)
       
        return (params9,magR)   

    # finish reading sensor data
    magnetometer._calibration_complete = True
    m = magnetometer._calibration_s[:,0:3]  # columns 0,1,2
    a = magnetometer._calibration_s[:,3:6]  # columns 3,4,5
    count = np.shape(magnetometer._calibration_s)[0]
    np.savetxt('magcal_data_raw_precision.csv', magnetometer._calibration_s)
    logging.info(f"Collected {count} samples for magnetometer calibration (check magcal_data_raw_precision.csv)")
    print(f"m {np.shape(m)}")
    print(f"a {np.shape(a)}")

    # average strength
    strength = 0
    for v in m:
        strength += np.linalg.norm(v) / count
    logging.info(f"Average strength of the samples is {strength} uT")

    # calculate the calibration matrix and offset
    m_mg = m * 10  # algorithm expects mG i.o. uT
    a_n = a / 9.81  # algorithm expects acceleration normalized to 1
    (params,magScale) = ellipsoid_iterate(m_mg,a_n,1)
    #(params,magScale) = ellipsoid_iterate_symmetric(m_mg,1)
    magnetometer.b = np.vstack(params[0:3]*magScale) / 10  # convert back to uT
    print(f"b {magnetometer.b}")
    magnetometer.A_1 = np.reshape(params[3:12],(3,3))
    print(f"A_1 {magnetometer.A_1}")

    mcal = np.dot(magnetometer.A_1, m.T - magnetometer.b).T
    np.savetxt('magcal_data_cal_precision.csv', mcal)

    graphic_report(m, mcal)

    # clear way for next calibration
    magnetometer._calibration_complete = None


def calibrate_magnetometer_with_old_data_teslabs():
    global magnetometer
    logging.info("Using calibration data in magcal_data_raw_teslabs.csv")
    magnetometer._calibration_s = np.loadtxt('magcal_data_raw_teslabs.csv')
    finish_calibration_teslabs()


def calibrate_magnetometer_with_old_data_precision():
    global magnetometer
    logging.info("Using calibration data in magcal_data_raw_precision.csv")
    magnetometer._calibration_s = np.loadtxt('magcal_data_raw_precision.csv')
    finish_calibration_precision()


def graphic_report(s, scal):
    fig1, ((yz, xz), (yx, dummy)) = plt.subplots(2,2)
    yz.set_xlabel('Y')
    yz.set_ylabel('Z')
    xz.set_xlabel('X')
    xz.set_ylabel('Z')
    yx.set_xlabel('Y')
    yx.set_ylabel('X')
    yz.set_aspect('equal')
    xz.set_aspect('equal')
    yx.set_aspect('equal')
    yz.set_xlim([-80,80])
    yz.set_ylim([-80,80])
    xz.set_xlim([-80,80])
    xz.set_ylim([-80,80])
    yx.set_xlim([-80,80])
    yx.set_ylim([-80,80])
    yz.plot(s[:,1], s[:,2], 'r.', ms=3)
    xz.plot(s[:,0], s[:,2], 'r.', ms=3)
    yx.plot(s[:,1], s[:,0], 'r.', ms=3)
    yz.plot(scal[:,1], scal[:,2], 'b.', ms=3)
    xz.plot(scal[:,0], scal[:,2], 'b.', ms=3)
    yx.plot(scal[:,1], scal[:,0], 'b.', ms=3)
    def circle(r,phi):
       return r*np.cos(phi), r*np.sin(phi)
    phis=np.arange(0,6.28,0.01)
    r = 49.081
    c = np.array(circle(r, phis)).T
    yz.plot(c[:,0], c[:,1], c='b',ls='-' )
    xz.plot(c[:,0], c[:,1], c='b',ls='-' )
    yx.plot(c[:,0], c[:,1], c='b',ls='-' )
    fig1.savefig('magcal_data.pdf')



class FXAS21002c(object):
    ''' FXAS21002c Simple Driver for gyroscope sensor readings.
    '''

    def __init__(self):
        self.i2c = board.I2C()
        self.sensor = adafruit_fxas21002c.FXAS21002C(i2c)

    def __del__(self):
        pass

    def read(self):
        ''' Get a sample.

            Returns
            -------
            s : list
                The sample in radians/s, [x, y, z].
        '''

        x, y, z = self.sensor.gyroscope
        return [x, y, z]


class FXOS8700_ms(object):
    ''' FXOS8700 Simple Driver for magnetic sensor readings.
    '''

    def __init__(self):
        self.i2c = board.I2C()
        self.sensor = adafruit_fxos8700.FXOS8700(self.i2c)

    def __del__(self):
        pass

    def read(self, number_of_readouts=1):
        ''' Get a sample.

            Returns
            -------
            s : list
                The sample in uT, [x, y, z].
        '''

        xs = 0.0
        ys = 0.0
        zs = 0.0
        for i in range(number_of_readouts):
            x, y, z = self.sensor.magnetometer
            xs += x
            ys += y
            zs += z
        xm = xs / number_of_readouts
        ym = ys / number_of_readouts
        zm = zs / number_of_readouts
        return [xm, ym, zm]


class FXOS8700_as(object):
    ''' FXOS8700 Simple Driver for acceleration sensor readings.
    '''

    def __init__(self):
        self.i2c = board.I2C()
        self.sensor = adafruit_fxos8700.FXOS8700(self.i2c)

    def __del__(self):
        pass

    def read(self):
        ''' Get a sample.

            Returns
            -------
            s : list
                The sample in uT, [x, y, z].
        '''

        x, y, z = self.sensor.accelerometer
        return [x, y, z]



def calibrate_magnetometer_teslabs():
    calibrate_magnetometer_generic(launch_calibration_teslabs, finish_calibration_teslabs)

def calibrate_magnetometer_precision():
    calibrate_magnetometer_generic(launch_calibration_precision, finish_calibration_precision)


def calibrate_magnetometer_generic(launch_calibration_function, finish_calibration_function):
    global magnetometer, accelerometer
    launch_calibration_function()  # starts reading
    logging.info("Start magnetometer calibration")
    logging.info("Start fully retract tilt motor")
    #MOTOR_T.backward()
    time.sleep(MOTOR_TRAVEL_TIME)
    logging.info("Stop fully retract tilt motor")
    #MOTOR_T.stop()
    logging.info("Start fully retract swivel motor")
    #MOTOR_S.backward()
    time.sleep(MOTOR_TRAVEL_TIME)
    logging.info("Stop fully retract swivel motor")
    #MOTOR_S.stop()
    forward = True
    remaining_amotor_travel_time = MOTOR_TRAVEL_TIME
    if NUMBER_OF_SCANS > 1:
        partial_amotor_travel_time = MOTOR_TRAVEL_TIME / (NUMBER_OF_SCANS - 1)
    while remaining_amotor_travel_time >= 0:
        if forward:
            logging.info("Start fully extend swivel motor")
            #MOTOR_S.forward()
            time.sleep(MOTOR_TRAVEL_TIME)
            logging.info("Stop fully extend swivel motor")
            #MOTOR_S.stop()
        else:
            logging.info("Start fully retract swivel motor")
            #MOTOR_S.backward()
            time.sleep(MOTOR_TRAVEL_TIME)
            logging.info("Stop fully retract swivel motor")
            #MOTOR_S.stop()
        if NUMBER_OF_SCANS == 1:
            break
        logging.info("Start partially extend tilt motor")
        #MOTOR_T.forward()
        time.sleep(partial_amotor_travel_time)
        #MOTOR_T.stop()
        logging.info("Stop partially extend tilt motor")
        remaining_amotor_travel_time -= partial_amotor_travel_time
    logging.info("Start fully retract tilt motor")
    #MOTOR_T.backward()
    time.sleep(MOTOR_TRAVEL_TIME)
    logging.info("Stop fully retract tilt motor")
    logging.info("Finish magnetometer calibration")
    finish_calibration_function()  # stops reading
    logging.info("Finished magnetometer calibration")
    

def save_magnetometer_calibration():
    global magnetometer
    magnetometer.save_calibration()


def load_magnetometer_calibration():
    global magnetometer
    magnetometer.load_calibration()


def print_magnetometer_calibration():
    global magnetometer
    print(magnetometer.A_1)
    print(magnetometer.b)


def swipe(duration=MOTOR_TRAVEL_TIME, direction=True, tstart=0):
    # read sensor values for specified duration and return them with timestamps
    global magnetometer, accelerometer
    vs = []
    ts = []
    start_time = time.time()
    while True:
        m = magnetometer.read_raw()  # sample in uT as np.array
        a = accelerometer.read_ms2()  # sample as np.array
        vs.append((m[0], m[1], m[2], a[0], a[1], a[2]))
        lapse_time = time.time() - start_time
        ts.append(lapse_time)
        if lapse_time >= duration: 
            break
        time.sleep(0.1)
    logging.info(f"Performed a {duration}s swipe {'forward' if direction else 'backward'} collecting {len(ts)} samples")
    return ts, vs  # python list, python list of 6-tuples

def find_tail(xs, threshold):
    # starting from the end, find the first value where change > threshold
    i = len(xs) - 1
    x = xs[i]
    while abs(x - xs[i]) < threshold:
        i--
        if i == 0:
            logging.error(f"No tail found")
            break
    logging.info(f"Tail found at value {i} of {len(xs)}")
    return i + 1

def trim_tail(vs):
    # figure out when the values stabilize and return the index and the trimmed vectors
    m_threshold = 2.0
    a_threshold = 0.2
    # reduce noisiness 
    vs = np.array(vs)
    l = len(vs)
    mxs = signal.savgol_filter(vs[:,0], 481, 2)
    mys = signal.savgol_filter(vs[:,1], 481, 2)
    mzs = signal.savgol_filter(vs[:,2], 481, 2)
    axs = signal.savgol_filter(vs[:,3], 481, 2)
    ays = signal.savgol_filter(vs[:,4], 481, 2)
    azs = signal.savgol_filter(vs[:,5], 481, 2)
    tails = []
    tails.append(find_tail(mxs, m_threshold))
    tails.append(find_tail(mys, m_threshold))
    tails.append(find_tail(mzs, m_threshold))
    tails.append(find_tail(axs, a_threshold))
    tails.append(find_tail(ays, a_threshold))
    tails.append(find_tail(azs, a_threshold))
    tails = np.extract(tails > l/2, tails)
    logging.info(f"From 6 tails, {len(tails)} were valid")
    tail = int(sum(tails)/len(tails))
    logging.info(f"Average tail at value {tail} of {l}")
    return tail, np.vstack((mxs[0:tail], mys[0:tail], mzs[0:tail], axs[0:tail], ays[0:tail], azs[0:tail])).T


def normalized_offset_to_hinge_angle_function():
    # generate an interpolation formula to convert a normalized offset (0-15cm scaled to 0-1) to an angle (0-pi/2 rads)
    q1 = 37.90
    q2 = 11.98
    ang = np.linspace(0,np.pi/2,90)
    off = np.sqrt((1/np.tan(ang/2)+q1)**2 + (1/np.tan(ang/2)+q2)**2 - 2*(1/np.tan(ang/2)+q1)*(1/np.tan(ang/2)+q2)*np.cos(ang))
    return interpolate.interp1d(off, ang)


def calibrate():
    global cal
    global magnetometer, accelerometer
    normalized_offset_to_hinge_angle = normalized_offset_to_hinged_angle_function()
    logging.info("Start heliostat position calibration")
    logging.info("Start fully retract tilt motor")
    #MOTOR_T.backward()
    time.sleep(MOTOR_TRAVEL_TIME)
    logging.info("Stop fully retract tilt motor")
    #MOTOR_T.stop()
    logging.info("Start fully retract swivel motor")
    #MOTOR_S.backward()
    time.sleep(MOTOR_TRAVEL_TIME)
    logging.info("Stop fully retract swivel motor")
    #MOTOR_S.stop()

    # swipes with constant tilt angle

    forward = True
    remaining_tilt_motor_travel_time = MOTOR_TRAVEL_TIME
    if NUMBER_OF_SCANS > 1:
        partial_tilt_motor_travel_time = MOTOR_TRAVEL_TIME / (NUMBER_OF_SCANS - 1)
    while remaining_tilt_motor_travel_time >= 0:
        if forward:
            logging.info("Start fully extend swivel motor")
            #MOTOR_S.forward()
            ts, vs = swipe(duration=MOTOR_TRAVEL_TIME, direction=forward)
            tmax, vs = trim_tail(vs)
            ts = normalized_offset_to_hinge_angle(ts/tmax)
            logging.info("Stop fully extend swivel motor")
            #MOTOR_S.stop()
        else:
            logging.info("Start fully retract swivel motor")
            #MOTOR_S.backward()
            ts, vs = swipe(duration=MOTOR_TRAVEL_TIME, direction=forward)
            # swiping backward, so flipping order of samples
            ts = np.flip(ts)  # TODO
            vs = np.flip(vs, 0)
            tmax, vs = trim_tail(vs)
            ts = normalized_offset_to_hinge_angle(ts/tmax)
            logging.info("Stop fully retract swivel motor")
            #MOTOR_S.stop()
        forward = not forward
        if NUMBER_OF_SCANS == 1:
            break
        logging.info("Start partially extend tilt motor")
        #MOTOR_T.forward()
        time.sleep(partial_tilt_motor_travel_time)
        logging.info("Stop partially extend tilt motor")
        #MOTOR_T.stop()
        remaining_tilt_motor_travel_time -= partial_tilt_motor_travel_time
    if not forward:
        logging.info("Start fully retract swivel motor")
        #MOTOR_S.backward()
        time.sleep(MOTOR_TRAVEL_TIME)
        logging.info("Stop fully retract swivel motor")
        #MOTOR_S.stop()
    logging.info("Start fully retract tilt motor")
    #MOTOR_T.backward()
    time.sleep(MOTOR_TRAVEL_TIME)
    logging.info("Stop fully retract tilt motor")

    # swipes with constant swivel angle

    forward = True
    remaining_swivel_motor_travel_time = MOTOR_TRAVEL_TIME
    if NUMBER_OF_SCANS > 1:
        partial_swivel_motor_travel_time = MOTOR_TRAVEL_TIME / (NUMBER_OF_SCANS - 1)
    while remaining_swivel_motor_travel_time >= 0:
        if forward:
            logging.info("Start fully extend swivel motor")
            #MOTOR_S.forward()
            ts, vs = swipe(duration=MOTOR_TRAVEL_TIME, direction=forward)
            tmax, vs = trim_tail(vs)
            ts = normalized_offset_to_hinge_angle(ts/tmax)
            logging.info("Stop fully extend swivel motor")
            #MOTOR_S.stop()
        else:
            logging.info("Start fully retract swivel motor")
            #MOTOR_S.backward()
            ts, vs = swipe(duration=MOTOR_TRAVEL_TIME, direction=forward)
            tmax, vs = trim_tail(vs)
            ts = normalized_offset_to_hinge_angle(ts/tmax)
            logging.info("Stop fully retract swivel motor")
            #MOTOR_S.stop()
        if NUMBER_OF_SCANS == 1:
            break
        logging.info("Start partially extend tilt motor")
        #MOTOR_T.forward()
        time.sleep(partial_swivel_motor_travel_time)
        logging.info("Stop partially extend tilt motor")
        #MOTOR_T.stop()
        remaining_swivel_motor_travel_time -= partial_swivel_motor_travel_time
    if not forward:
        logging.info("Start fully retract swivel motor")
        #MOTOR_S.backward()
        time.sleep(MOTOR_TRAVEL_TIME)
        logging.info("Stop fully retract swivel motor")
        #MOTOR_S.stop()
    logging.info("Start fully retract tilt motor")
    #MOTOR_T.backward()
    time.sleep(MOTOR_TRAVEL_TIME)
    logging.info("Stop fully retract tilt motor")
    logging.info("Finish magnetometer calibration")
    logging.info("Finished heliostat position calibration")


def save_calibration():
    global cal
    logging.info("Writing data to file")
    with open('tcxss.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(range(len(cal['tcxss'])))
        for tcxs in zip(*cal['tcxss']):
            writer.writerow(tcxs)
    with open('tcyss.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(range(len(cal['tcyss'])))
        for tcys in zip(*cal['tcyss']):
            writer.writerow(tcys)
    with open('tczss.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(range(len(cal['tczss'])))
        for tczs in zip(*cal['tczss']):
            writer.writerow(tczs)
    with open('acxyzs.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['x','y','z'])
        for acxyz in zip(cal['acxs'], cal['acys'], cal['aczs']):
            writer.writerow(acxyz)


def load_calibration():
    global cal
    with open('tcxss.csv', 'r') as f:
        reader = csv.reader(f)
        next(reader)
        cal['tcxss'] = [list(map(float, tcxs)) for tcxs in zip(*reader)]
    with open('tcyss.csv', 'r') as f:
        reader = csv.reader(f)
        next(reader)
        cal['tcyss'] = [list(map(float, tcys)) for tcys in zip(*reader)]
    with open('tczss.csv', 'r') as f:
        reader = csv.reader(f)
        next(reader)
        cal['tczss'] = [list(map(float, tczs)) for tczs in zip(*reader)]
    with open('acxyzs.csv', 'r') as f:
        reader = csv.reader(f)
        next(reader)
        acxyzs = list(zip(*reader))
        cal['acxs'] = list(map(float, acxyzs[0]))
        cal['acys'] = list(map(float, acxyzs[1]))
        cal['aczs'] = list(map(float, acxyzs[2]))


def projection_on_vector(v1, v2):
    ''' Returns vector projection of v1 on v2
    '''
    return (np.dot(v1, v2) / np.dot(v2, v2)) * v2


def projection_on_plane(v1, n):
    ''' Returns project of v1 on the plane defined by normal vector n
    '''
    return v1 - projection_on_vector(v1, n)


def angle_between_vectors(v1, v2):
    ''' Returns the angle between v1 and v2 in radians
    '''
    v1_unit = v1 / np.linalg.norm(v1)
    v2_unit = v2 / np.linalg.norm(v2)
    return np.arccos(np.dot(v1_unit, v2_unit))


def angle_between_vector_and_plane(v, n):
    '''Returns the angle between v and the plane defined by it's normal vector n
    '''
    vp = projection_on_plane(v, n)
    return angle_between_vectors(v, vp)


def track_inclination_and_heading():
    global accelerometer, magnetometer
    try:
        while True:
            a = accelerometer.read()
            m = magnetometer.read_cal()
            vertical = np.array([-1,0,0])  # X-axis is vertical pointing up when collapsed; depends on how the sensor is mounted!
            # elevation is the angle between a and vertical
            elevation = np.degrees(angle_between_vectors(a, vertical))
            # normal vector of the mirror
            mirror_normal = np.array([0,1,0])  # Y-axis is perpendicular to the mirror; depends on how the sensor is mounted!
            # projection of the Y-axis on the horizontal plane
            y_horizontal = projection_on_plane(mirror_normal, a)
            # projection of the magnetic vector on the horizontal plane
            m_horizontal = projection_on_plane(m, a)
            # heading is angle between these two projected vectors in the horizontal plane
            heading = np.degrees(angle_between_vectors(y_horizontal, m_horizontal))
            # magnetic inclincation is the angle between m and m_horizontal
            magnetic_inclination = np.degrees(angle_between_vectors(m, m_horizontal))
            print("elevation: {:4.1f} - heading: {:4.1f} - magnetic inclination: {:4.1f}".format(elevation, heading, magnetic_inclination), end="\r")
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass


def track_madgwick():
    global accelerometer, magnetometer, gyroscope
    madgwick = Madgwick()
    try:
        time_previous = time.time()
        q_previous = np.array([1.0, 0.0, 0.0, 0.0])
        while True:
            a = accelerometer.read_ms2()
            m = magnetometer.read_cal()  # in uT
            g = gyroscope.read_rads()
            time_now = time.time()
            time_delay = time_now - time_previous
            q_now = madgwick.updateMARG(q_previous, gyr=g, acc=a, mag=1000*m, dt=time_delay)  # algorithm wants nT i.o. uT
            time_previous = time_now
            q_previous = q_now
            e = Quaternion(q_now).to_angles()
            '''
            roll = e[0]
            pitch = e[1]
            yaw = e[2]
            v = [np.cos(yaw) * np.cos(pitch), np.sin(yaw) * np.cos(pitch), np.sin(pitch)]
            XY = np.array([0.0, 0.0, 1.0])
            elevation = angle_between_vector_and_plane(v, XY)
            X = np.array([1.0, 0.0, 0.0])
            heading = angle_between_vectors(projection_on_plane(v, XY), X)
            print("roll: {:4.1f} - pitch: {:4.1f} - yaw: {:4.1f} - heading: {:4.1f} - elevation: {:4.1f} - frequency: {:4.1f}".format(np.degrees(roll), np.degrees(pitch), np.degrees(yaw), np.degrees(heading), np.degrees(elevation), 1/time_delay), end="\r")
            '''
            print("a: {:6.1f},{:6.1f},{:6.1f} - m: {:6.1f},{:6.1f},{:6.1f} - g: {:6.1f},{:6.1f},{:6.1f} - roll: {:4.1f} - pitch: {:4.1f} - yaw: {:4.1f} - frequency: {:4.1f}".format(a[0], a[1], a[2], m[0], m[1], m[2], g[0], g[1], g[2], np.degrees(e[0]), np.degrees(e[1]), np.degrees(e[2]), 1/time_delay), end="\r")
    except KeyboardInterrupt:
        pass


def track_mahony():
    global accelerometer, magnetometer, gyroscope
    mahony = Mahony()
    try:
        time_previous = time.time()
        q_previous = np.array([1.0, 0.0, 0.0, 0.0])
        while True:
            a = accelerometer.read_ms2()
            m = magnetometer.read_cal()  # in uT
            g = gyroscope.read_rads()
            time_now = time.time()
            time_delay = time_now - time_previous
            q_now = mahony.updateMARG(q_previous, gyr=g, acc=a, mag=1000*m, dt=time_delay)  # algorithm wants nT i.o. uT
            time_previous = time_now
            q_previous = q_now
            e = Quaternion(q_now).to_angles()
            print("roll: {:4.1f} - pitch: {:4.1f} - yaw: {:4.1f} - frequency: {:4.1f}".format(np.degrees(e[0]), np.degrees(e[1]), np.degrees(e[2]), 1/time_delay), end="\r")
    except KeyboardInterrupt:
        pass


def track_magnetic():
    global magnetometer
    try:
        while True:
            m = magnetometer.read_cal()
            xy = np.array([0, 0, 1])  # normal vector of XY plane
            i = angle_between_vector_and_plane(m, xy)
            xz = np.array([0, 1, 0])  # normal vector of XZ plane
            h = angle_between_vector_and_plane(m, xz)
            a = np.linalg.norm(m)
            print("x,y,z: {:4.1f},{:4.1f},{:4.1f} - inclination (vs XY): {:4.1f} - heading (vs XZ): {:4.1f} - strength: {:8.1f}".format(m[0], m[1], m[2], np.degrees(i), np.degrees(h), a), end="\r")
    except KeyboardInterrupt:
        pass


def track_random():
    global cal
    acidx = random.randrange(0,len(cal['acxs']))
    tcidx = random.randrange(0,len(cal['tcxss'][0]))
    racx = cal['acxs'][acidx]
    racy = cal['acys'][acidx]
    racz = cal['aczs'][acidx]
    rtcx = cal['tcxss'][acidx][tcidx]
    rtcy = cal['tcyss'][acidx][tcidx]
    rtcz = cal['tczss'][acidx][tcidx]
    try:
        while True:
            tcx, tcy, tcz = sensor.magnetometer
            acx, acy, acz = sensor.accelerometer
            aangle = vg.angle(np.array([racx, racy, racz]), np.array([acx, acy, acz]))
            tangle = vg.angle(np.array([rtcx, rtcy, rtcz]), np.array([tcx, tcy, tcz]))
            print("elevation: {:4.1f} - heading: {:4.1f}".format(aangle, tangle), end="\r")
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass



def scan():
    logging.info("Start heading scan")
    tcxs = []
    tcys = []
    tczs = []
    stop = time.time() + MOTOR_TRAVEL_TIME  # seconds
    logging.info("Start full travel extending of swivel motor")
    MOTOR_S.forward()
    while time.time() < stop:
        tcx, tcy, tcz = sensor.magnetometer
        tcxs.append(tcx)
        tcys.append(tcy)
        tczs.append(tcz)
        time.sleep(0.1)
    logging.info("Stop full travel extending of swivel motor")
    MOTOR_S.stop()
    logging.info("Stop heading scan")
    return tcxs, tcys, tczs


def pyplot_demo():
    n = 1024
    X = np.random.normal(0, 1, n)
    Y = np.random.normal(0, 1, n)
    Z = np.random.normal(0, 1, n)

    fig1 = plt.figure(1)
    ax1 = fig1.add_subplot(111, projection='3d')
    ax1.scatter(X, Y, Z, s=5, color='r')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    fig1.savefig('pyplot_demo.pdf')


def main():
    global accelerometer, magnetometer, gyroscope
    accelerometer = Accelerometer()
    magnetometer = Magnetometer()
    gyroscope = Gyroscope()
    options = [
        ['Calibrate magnetometer (juddzone precision asymmetric)', calibrate_magnetometer_precision],
        ['Calibrate magnetometer with old data (juddzone precision asymmetric)', calibrate_magnetometer_with_old_data_precision],
        ['Calibrate magnetometer (teslabs)', calibrate_magnetometer_teslabs],
        ['Calibrate magnetometer with old data (teslabs)', calibrate_magnetometer_with_old_data_teslabs],
        ['Save magnetometer calibration to file', save_magnetometer_calibration],
        ['Load magnetometer calibration from file', load_magnetometer_calibration],
        ['Print magnetometer calibration', print_magnetometer_calibration],
        ['Calibrate', calibrate],
        ['Save calibration to file', save_calibration],
        ['Load calibration from file', load_calibration],
        ['Track angular distance to random calibration point', track_random],
        ['Track inclination and heading angles', track_inclination_and_heading],
        ['Track Madgwick', track_madgwick],
        ['Track Mahony', track_mahony],
        ['Track magnetic sensor', track_magnetic],
        ['Pyplot demo', pyplot_demo]
    ]
    terminal_menu = TerminalMenu([option[0] for option in options])
    while True:
        choice = terminal_menu.show()
        if type(choice) != int:
            break
        options[choice][1]()  # run the function stored as second element in the chosen row of options


if __name__ == "__main__":
    main()

