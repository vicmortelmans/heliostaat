#!/usr/bin/python3

# Heliostat driver - this script should be run as a service

# In the end, this script should provide the remote IoT interface

# During development, it's invoked on command line

# Run with bokeh:
#   bokeh serve --allow-websocket-origin=nymea.local:5006 helio.py
# Then open the website to continue the script

import adafruit_fxos8700
import adafruit_fxas21002c
from ahrs import Quaternion
from ahrs.filters import Madgwick, Mahony
import board
from bokeh.plotting import curdoc, figure
from bokeh.models import Button
import copy
import csv
from gpiozero import Motor
import logging
import numpy as np
import random
from scipy import linalg, signal, interpolate
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
import scipy.ndimage
import sys
import threading
import time
import vg

'''
ribs data structure:
[
    {
        'motorname': 'swivel' or 'tilt',  # 'swivel' means: constant tilt angle, and vv.
        'start': start index in cal['vs'] and derived arrays
        'length': number of entries in cal['vs'] and derived arrays
        'tmp': {'i1', 'i2', 'd1', 'd2', 'd'}  # while interpolating, contains data about closest points on this rib
    }
]
'''

handler_lock = threading.Semaphore(0)  # handler waiting for data
generator_lock = threading.Semaphore(0)  # generator waiting for user pushing the button

# Setup logging
logging.basicConfig(force=True, format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d %(funcName)s] %(message)s', datefmt='%Y-%m-%d:%H:%M:%S', level=logging.INFO)
# force=True is needed because bokeh sets the level down

# Setup motors
MOTOR_TRAVEL_TIME = 25  # 25 seconds, 10 for speeding up durig debugging
MOTOR_S = Motor(26, 20)
MOTOR_T = Motor(19, 16)

# Setup calibration data global variable
NUMBER_OF_PARTIAL_SWIPES = 5
DOWNSAMPLING = 20  # 4000 samples => 20 samples


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

    def read_raw(self):
        ''' Get a sample.

            Returns
            -------
            s : np.array
                The reading in uT (not corrected).
        '''
        return np.array(self.sensor.read())

    def read_average(self, number_of_readouts):
        ''' Get 10 samples and take average.

            Returns
            -------
            s : np.array
                The reading in uT (not corrected).
        '''
        return np.array(self.sensor.read(number_of_readouts=number_of_readouts))


class FXAS21002c(object):
    ''' FXAS21002c Simple Driver for gyroscope sensor readings.
    '''

    def __init__(self):
        self.i2c = board.I2C()
        self.sensor = adafruit_fxas21002c.FXAS21002C(self.i2c)

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
            if number_of_readouts > 1:
                time.sleep(0.001)
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

def read_level():
    # This function assumes both motors to be fully retracted !!
    global accelerometer
    a = accelerometer.read_ms2()
    # level is angle between gravity vector and YZ plane
    level = angle_between_vector_and_plane(a, np.array([1,0,0]))
    # helio_tilt is angle between gravity vector and XZ plane
    helio_tilt = angle_between_vector_and_plane(a, np.array([0,1,0]))
    return level, helio_tilt

def scan(duration=MOTOR_TRAVEL_TIME):
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
        #plot(data={'x': [lapse_time, lapse_time, lapse_time], 'y': [m[0], m[1], m[2]]}, title="scan"))
        #time.sleep(0.01)
    logging.info(f"Performed a {duration}s scan collecting {len(ts)} samples")
    return np.array(ts), np.array(vs)  # array, array of 6 columns

def find_tail(xs, threshold):
    minimum = np.min(xs)
    maximum = np.max(xs)
    if maximum - minimum < 300 * threshold:
        logging.warning(f"Tail finding with threshold {threshold} in data with range {maximum - minimum} doesn't make sense")
        return 0
    else:
    # starting from the end, find the first value where change > threshold
        length = len(xs) - 1
        i = length
        x = xs[i]
        running_min = x
        running_max = x
        while running_max - running_min < threshold:
            i = i-1
            x = xs[i]
            if i == 0:
                logging.error(f"No tail found")
                return 0
            else:
                running_min = min(running_min, x)
                running_max = max(running_max, x)
        index = i + 1
        logging.info(f"Tail found at index {i} ({int(i/length*100)}%) based on threshold {threshold} in data with range {maximum - minimum}")
        return index

def trim_tail(vs):
    # figure out when the values stabilize (only using acceleration, much stabler signal!) and
    # return the index and the trimmed and smoothened vectors
    a_threshold = 0.003  # established based on visual inspection of readout graphs
    # reduce noisiness 
    vs = np.array(vs)
    l = len(vs)
    # smoothen the readings
    mxs = scipy.ndimage.uniform_filter1d(vs[:,0], 500)
    mys = scipy.ndimage.uniform_filter1d(vs[:,1], 500)
    mzs = scipy.ndimage.uniform_filter1d(vs[:,2], 500)
    axs = scipy.ndimage.uniform_filter1d(vs[:,3], 500)
    ays = scipy.ndimage.uniform_filter1d(vs[:,4], 500)
    azs = scipy.ndimage.uniform_filter1d(vs[:,5], 500)
    logging.info("Finding tails for resp. ax, ay, az")
    length = len(axs)
    plot(data={'x': np.hstack((np.arange(length),np.arange(length),np.arange(length))), 'y': np.hstack((axs,ays,azs))}, title="as")
    #plot(data={'x': np.arange(length), 'y': axs}, title="axs")
    #plot(data={'x': np.arange(length), 'y': ays}, title="ays")
    #plot(data={'x': np.arange(length), 'y': azs}, title="azs")
    tails = []
    tails.append(find_tail(axs, a_threshold))
    tails.append(find_tail(ays, a_threshold))
    tails.append(find_tail(azs, a_threshold))
    # for sanity, check if tails > 1/2
    tails = np.extract(np.greater(tails, l/2), tails)
    logging.info(f"From 3 tails, {len(tails)} were valid")
    if tails.size == 0:
        logging.error("No valid tails found, readouts are probably flat, no trimming")
        tail = l
    else:
        tail = int(sum(tails)/len(tails))
        logging.info(f"Average tail at value {tail} of {l}")
    return tail, np.vstack((mxs[0:tail], mys[0:tail], mzs[0:tail], axs[0:tail], ays[0:tail], azs[0:tail])).T


def normalized_offset_to_hinge_angle_function():
    # generate an interpolation formula to convert a normalized offset (0-15cm scaled to 0-1) to an angle (0-pi/2 rads)
    q1 = 37.90
    q2 = 11.98
    ang = np.linspace(0.001,np.pi/2,90)
    off = (np.sqrt((1/np.tan(ang/2)+q1)**2 + (1/np.tan(ang/2)+q2)**2 - 2*(1/np.tan(ang/2)+q1)*(1/np.tan(ang/2)+q2)*np.cos(ang)) - 26) / 15  # 15 is the movement range of the motor in cm, 26 is the minimum length
    ang = np.hstack((0,ang))
    off = np.hstack((0,off))
    #import pdb; pdb.set_trace()
    return interpolate.interp1d(off, ang)

def downsample_columns(vs, new_nrows):
    ncols = vs.shape[1]
    nrows = vs.shape[0]
    new_vs = np.empty([new_nrows, ncols])
    for col in range(ncols):
        interpolation = interpolate.interp1d(np.arange(nrows),vs[:,col])
        new_vs[:,col] = interpolation(np.linspace(0, nrows-1, new_nrows))
    return new_vs

def downsample_array(a, new_n):
    n = len(a)
    interpolation = interpolate.interp1d(np.arange(n),a)
    new_a = interpolation(np.linspace(0, n-1, new_n))
    return new_a

def process_swipe(times, vs, special_idx, forward=True):
    # trim the readouts to where the movement stops (= 90 degrees)
    # and convert times to angles
    # special_idx is a list of indices, for which a list of corresponding angles is returned (only forward?)
    normalized_offset_to_hinge_angle = normalized_offset_to_hinge_angle_function()
    timeindexmax, vs = trim_tail(vs) 
    times = times[0:timeindexmax]
    timemax = times[-1]  # last element = times[timeindexmax-1]
    if not forward:
        logging.info("swiping backward, so flipping order of timestamps, because offset_to_hinge function is direction-dependent and fist v should map to 90 degrees and last v to zero degrees!")
        times = np.flip(times)
        #vs = np.flip(vs, 0)
    angs = normalized_offset_to_hinge_angle(times/timemax)
    # special angles
    if special_idx:
        special_angles = [normalized_offset_to_hinge_angle(times[i]/timemax) for i in special_idx if i < timeindexmax]
    else:
        special_angles = None
    # downsample
    number_of_samples = DOWNSAMPLING
    vs = downsample_columns(vs, number_of_samples)
    angs = downsample_array(angs, number_of_samples)
    if special_idx:
        return angs, vs, special_angles
    else:
        return angs, vs
    
def swipe(forward=None, MOTOR=None, motorname=None):
    logging.info(f"Start fully {'extending' if forward else 'retracting'} {motorname} motor")
    MOTOR.forward() if forward else MOTOR.backward()
    times, vs = scan(duration=MOTOR_TRAVEL_TIME)
    logging.info(f"Stop fully {'extending' if forward else 'retracting'} {motorname} motor")
    MOTOR.stop()
    return times, vs

def partial_swipe(forward=None, duration=None, MOTOR=None, motorname=None):
    logging.info(f"Start partially {'extending' if forward else 'retracting'} {motorname} motor")
    MOTOR.forward() if forward else MOTOR.backward()
    times, vs = scan(duration=duration)
    logging.info(f"Stop partially {'extending' if forward else 'retracting'} {motorname} motor")
    MOTOR.stop()
    return times, vs

def retract(MOTOR=None, motorname=None):
    logging.info(f"Start fully retract {motorname} motor")
    MOTOR.backward()
    time.sleep(MOTOR_TRAVEL_TIME)
    logging.info(f"Stop fully retract {motorname} motor")
    MOTOR.stop()

def retract_motors():
    retract(MOTOR=MOTOR_S, motorname="sweep")
    retract(MOTOR=MOTOR_T, motorname="tilt")

def register(ss, ts, vs, motorname):
    global cal
    logging.info(f"Registering {len(ss)} samples")
    if 'ss' not in cal:
        # first entry
        cal['ribs'] = [{
            'motorname': motorname,
            'start': 0,
            'length': len(ss)
        }]
        cal['ss'] = ss
        cal['ts'] = ts
        cal['vs'] = vs
    else:
        cal['ribs'].append({
            'motorname': motorname,
            'start': len(cal['ss']),  # if cal['ss'] already has 20 elements, the new rib starts at 20
            'length': len(ss)
        }
        cal['ss'] = np.hstack((cal['ss'], ss))
        cal['ts'] = np.hstack((cal['ts'], ts))
        cal['vs'] = np.vstack((cal['vs'], vs))

def register_position(level, helio_tilt):
    global cal
    cal['level'] = level
    cal['helio_tilt'] = helio_tilt


def register_by_name(array, name):
    global cal
    logging.info(f"Registering {len(array)} samples as {name}")
    cal[name] = array

def normal_vectors():
    global cal
    ht = cal['helio_tilt']  # readibility
    for i in range(len(cal['vs'])):
        s = cal['ss'][i]
        t = cal['ts'][i]
        x = -np.sin(s)*np.cos(t)
        y = np.cos(t)*np.cos(s)*np.cos(ht) - np.sin(t)*np.sin(ht)
        z = np.cos(t)*np.cos(s)*np.sin(ht) + np.sin(t)*np.cos(ht)
        if i == 0:
            ns = np.array([x, y, z])
        else:
            ns = np.vstack((ns, [x, y, z]))
    return ns

def elevations():
    global cal
    for i in range(len(cal['vs'])):
        # elevation is angle between normal vector of the mirror and the XY plane
        e = angle_between_vector_and_plane(cal['ns'][i], [0,0,1])
        if i == 0:
            es = np.array([e])
        else:
            es = np.hstack((es, e))
    return es

def headings():
    global cal
    for i in range(len(cal['vs'])):
        # heading is angle between normal vector of the mirror and the XZ plane
        h = angle_between_vector_and_plane(cal['ns'][i], [0,1,0])
        if i == 0:
            hs = np.array([h])
        else:
            hs = np.hstack((hs, h))
    return hs

# THIS IS A BOKEH CALLBACK:
def data_handler():
    global plot_data
    global plot_title
    global plot_final
    if __name__.startswith('bokeh'):
        # wait until data is ready
        logging.info("Handler: waiting for data from generator")
        handler_lock.acquire()
        logging.info("Handler: handling data from generator")
    # add new graph
    p = figure(title=plot_title)
    plot_circle = p.circle([],[])
    curdoc().add_root(p)
    plot_circle.data_source.stream(plot_data)
    # add the button for the user to trigger the callback that will draw the next graph
    if not plot_final:
        button = Button(label="Continue data generation", button_type="success")
        button.on_click(data_handler)
        curdoc().add_root(button)
    if __name__.startswith('bokeh'):
        # unlock the generator to continue generating data
        logging.info("Handler: unlocking generator")
        generator_lock.release()
    # actual drawing is only done after the callback finishes!

def ref_col(a):
    # select from array with 6 columns mx, my, mz, ax, ay, az, the 3 columns that will be reference
    # for interpolation, currently: mx, ax, ay
    if a.ndim > 1:
        return a[:,[0,3,4]]
    else:
        return a[[0,3,4]]

def calibrate():
    global cal
    global elevation  # function
    global heading  # function

    cal = {}  # reset global variable containing the calibration values

    if NUMBER_OF_PARTIAL_SWIPES > 1:
        partial_motor_travel_time = MOTOR_TRAVEL_TIME / NUMBER_OF_PARTIAL_SWIPES
    else:
        partial_motor_travel_time = MOTOR_TRAVEL_TIME

    logging.info("Start heliostat position calibration")
    #retract(MOTOR=MOTOR_T, motorname="tilt")
    #retract(MOTOR=MOTOR_S, motorname="swivel")

    # calculate the tilt angle of the whole heliostat
    level, helio_tilt = read_level()
    register_position(level, helio_tilt)
    if level > np.pi/20:
        logging.warning(f"The heliostat isn't mounted level, it's {np.degrees(level)} degrees off.")
    logging.info(f"The heliostat is mounted with a tilt of {np.degrees(helio_tilt)} degrees.")

    # sequence of forward partial swivel swipes with 0 tilt angle #0-19

    for i in range(NUMBER_OF_PARTIAL_SWIPES):
        logging.info(f"Forward partial swivel swipe #{i} with 0 tilt angle")
        if i > 0: time.sleep(2)
        partial_times, partial_vs = partial_swipe(duration=partial_motor_travel_time, forward=True, MOTOR=MOTOR_S, motorname="swivel")
        if i == 0:
            times = partial_times
            vs = partial_vs
            swipe_idxs = [len(times) - 1]  # list of indices of end times for each partial swipe
        else:
            times = np.hstack((times, partial_times + times[-1]))  # starting from end time in previous partial swipe
            vs = np.vstack((vs, partial_vs))
            swipe_idxs.append(len(times))  
    ss, vs, s_angles = process_swipe(times, vs, swipe_idxs, forward=True)
    register(ss, np.full((1, len(ss)), 0)[0], vs, "swivel")  # ts are all zero, full returns [[]], so I need [0]
    logging.info(f"Swipe angles are {np.degrees(s_angles)} based on indices {swipe_idxs}")

    # sequence of forward partial tilt swipes with 90 swivel angle #20-39

    for i in range(NUMBER_OF_PARTIAL_SWIPES):
        logging.info(f"Forward partial tilt swipe #{i} with 90 swivel angle")
        if i > 0: time.sleep(2)
        partial_times, partial_vs = partial_swipe(duration=partial_motor_travel_time, forward=True, MOTOR=MOTOR_T, motorname="tilt")
        if i == 0:
            times = partial_times
            vs = partial_vs
            swipe_idxs = [len(times) - 1]  # list of indices of end times for each partial swipe
        else:
            times = np.hstack((times, partial_times + times[-1]))  # starting from end time in previous partial swipe
            vs = np.vstack((vs, partial_vs))
            swipe_idxs.append(len(times))  
    ts, vs, t_angles = process_swipe(times, vs, swipe_idxs, forward=True)
    register(np.full((1, len(ts)), np.pi/2)[0], ts, vs, "tilt")  # ss are all 90 degrees
    logging.info(f"Tilt angles are {np.degrees(t_angles)} based on indices {swipe_idxs}")

    # sequence of backward partial swivel swipes with 90 tilt angle #40-59

    for i in range(NUMBER_OF_PARTIAL_SWIPES):
        logging.info(f"Backward partial swivel swipe #{i} with 90 tilt angle")
        partial_times, partial_vs = partial_swipe(duration=partial_motor_travel_time, forward=False, MOTOR=MOTOR_S, motorname="swivel")
        if i == 0:
            times = partial_times
            vs = partial_vs
        else:
            time.sleep(1)
            times = np.hstack((times, partial_times + times[-1]))  # starting from end time in previous partial swipe
            vs = np.vstack((vs, partial_vs))
    ss, vs = process_swipe(times, vs, None, forward=False)
    register(ss, np.full((1, len(ss)), np.pi/2)[0], vs, "swivel")  # ts are all 90 degrees

    # sequence of backward partial tilt swipes with 0 swivel angle #60-79

    for i in range(NUMBER_OF_PARTIAL_SWIPES):
        logging.info(f"Backward partial tilt swipe #{i} with 0 swivel angle")
        partial_times, partial_vs = partial_swipe(duration=partial_motor_travel_time, forward=False, MOTOR=MOTOR_T, motorname="tilt")
        if i == 0:
            times = partial_times
            vs = partial_vs
        else:
            time.sleep(1)
            times = np.hstack((times, partial_times + times[-1]))  # starting from end time in previous partial swipe
            vs = np.vstack((vs, partial_vs))
    ts, vs = process_swipe(times, vs, None, forward=False)
    register(np.full((1, len(ts)), 0)[0], ts, vs, "tilt")  # ss are all 0

    # swipes with constant tilt angle #80-99, 100-119, 120-139, 140-159

    forward = True
    for t in t_angles:  # skipping the first scan at 0 degrees
        logging.info(f"Swivel swipe with {np.degrees(t)} tilt angle")
        partial_swipe(duration=partial_motor_travel_time, forward=True, MOTOR=MOTOR_T, motorname="tilt")
        times, vs = swipe(forward=forward, MOTOR=MOTOR_S, motorname="swivel")
        ss, vs = process_swipe(times, vs, None, forward=forward)
        register(ss, np.full((1, len(ss)), t)[0], vs, "swivel")  # ts are all t (from t_angles)
        forward = not forward  # reverse direction for next iteration
    if not forward:  # last swipe was forward
        retract(MOTOR=MOTOR_S, motorname="swivel")
    retract(MOTOR=MOTOR_T, motorname="tilt")

    # swipes with constant swivel angle #160-179, 180-199, 200-219, 220-239

    forward = True
    for s in s_angles:
        logging.info(f"Tilt swipe with {np.degrees(s)} swivel angle")
        partial_swipe(duration=partial_motor_travel_time, forward=True, MOTOR=MOTOR_S, motorname="swivel")
        times, vs = swipe(forward=forward, MOTOR=MOTOR_T, motorname="tilt")
        ts, vs = process_swipe(times, vs, None, forward=forward)
        register(np.full((1, len(ts)), s)[0], ts, vs, "tilt")  # ss are all s (from s_angles)
        forward = not forward  # reverse direction for next iteration
    if not forward:  # last swipe was forward
        retract(MOTOR=MOTOR_T, motorname="tilt")
    retract(MOTOR=MOTOR_S, motorname="swivel")

    # calculate for each sample the absolute normal vector of the mirror
    # (in coordinate system with vertical Z, thus including the effect of the helio tilt)
    logging.info("Register ns")
    register_by_name(normal_vectors(), 'ns')
    # calculate for each sample the mirror elevation and heading
    logging.info("Register es")
    register_by_name(elevations(), 'es')
    logging.info("Register hs")
    register_by_name(headings(), 'hs')

    # initialize interpolcation functions
    logging.info("Generate elevation interpolator")
    #elevation = LinearNDInterpolator(cal['vs'], cal['es'])
    elevation = LinearNDInterpolator(ref_col(cal['vs']), cal['es'])
    logging.info("Generate heading interpolator")
    #heading = LinearNDInterpolator(cal['vs'], cal['hs'])
    heading = LinearNDInterpolator(ref_col(cal['vs']), cal['hs'])

    logging.info("Finished heliostat position calibration")


def save_calibration():
    global cal
    logging.info("Writing data to file")
    np.savetxt('ss.csv', cal['ss'])
    np.savetxt('ts.csv', cal['ts'])
    np.savetxt('vs.csv', cal['vs'])
    np.savetxt('ns.csv', cal['ns'])
    np.savetxt('es.csv', cal['es'])
    np.savetxt('hs.csv', cal['hs'])
    np.savetxt('pos.csv', np.array([cal['level'], cal['helio_tilt']]))
    with open('ribs.json', 'w') as outfile:
        json.dump(cal['ribs'], outfile)


def load_calibration():
    global cal
    global elevation  # function
    global heading  # function
    cal = {}
    cal['ss'] = np.loadtxt('ss.csv')
    cal['ts'] = np.loadtxt('ts.csv')
    cal['vs'] = np.loadtxt('vs.csv')
    cal['ns'] = np.loadtxt('ns.csv')
    cal['es'] = np.loadtxt('es.csv')
    cal['hs'] = np.loadtxt('hs.csv')
    cal['level'], cal['helio_tilt'] = np.loadtxt('pos.csv')
    with open('ribs.json') as json_file:
        cal['ribs'] = json.load(json_file)
    # initialize interpolcation functions
    logging.info("Generate elevation interpolator")
    #elevation = LinearNDInterpolator(cal['vs'], cal['es'])
    elevation = LinearNDInterpolator(ref_col(cal['vs']), cal['es'])
    logging.info("Generate heading interpolator")
    #heading = LinearNDInterpolator(cal['vs'], cal['hs'])
    heading = LinearNDInterpolator(ref_col(cal['vs']), cal['hs'])

def rectangular_interpolator(vin):
    global cal
    vin = ref_col(vin)
    # for each rib, find the two closest points
    for rib in cal['ribs']:
        v1 = None; v2 = None; d1 = None; d2 = None; i1 = None; i2 = None
        # iterate the cal indices that compose this rib:
        for i in range(rib['start'], rib['start']+rib['length']):
            v = ref_col(cal['vs'][i])
            d = np.linalg.norm(v - vin)
            if not d1:
                d1 = d; v1 = v; i1 = i
            elif d <= d1:
                d2 = d1; v2 = v1; i2 = i1
                d1 = d; v1 = v; i1 = i
            elif not d2 or d < d2:
                d2 = d; v2 = v; i2 = i
        if abs(i1 - i2) > 1: 
            logging.warning(f"Two closest points on rib are not adjacent: {i1} and {i2}")
        d, d1, d2 = distance_point_to_line_and_projection_to_end_points(vin, v1, v2)   
        rib['tmp'] = {'i1': i1, 'i2': i2, 'd1': d1, 'd2': d2, 'd': d}
    # find closest swivel rib
    s1 = None
    for rib in cal['ribs']:
        if rib['motorname'] == 'swivel':
            if not s1 or rib['tmp']['d'] < s1['tmp']['d']:
                s1 = rib
    logging.debug(f"Closest swivel rib starts at {s1['start']}")
    # find second closes swivel rib (on opposite side!)
    s2 = None
    for rib in cal['ribs']:
        if rib['motorname'] == 'swivel' and not rib == s1:
            # test if the potential 'second close' point on this rib is on the opposite side of the input:
            if point_between_two_points_3d(vin, ref_col(cal['vs'][s1['tmp']['i1']]), ref_col(cal['vs'][rib['tmp']['i2']]):
                if not s2 or rib['tmp']['d'] < s2['tmp']['d']:
                    s2 = rib
    logging.debug(f"Second closest swivel rib starts at {s2['start']}")
    # find closest tilt rib
    t1 = None
    for rib in cal['ribs']:
        if rib['motorname'] == 'tilt':
            if not t1 or rib['tmp']['d'] < t1['tmp']['d']:
                t1 = rib
    logging.debug(f"Closest tilt rib starts at {t1['start']}")
    # find second closes tilt rib (on opposite side!)
    t2 = None
    for rib in cal['ribs']:
        if rib['motorname'] == 'tilt' and not rib == t1:
            # test if the potential 'second close' point on this rib is on the opposite side of the input:
            if point_between_two_points_3d(vin, ref_col(cal['vs'][t1['tmp']['i1']]), ref_col(cal['vs'][rib['tmp']['i2']]):
                if not t2 or rib['tmp']['d'] < t2['tmp']['d']:
                    t2 = rib
    logging.debug(f"Second closest tilt rib starts at {t2['start']}")
    # interpolate elevation and heading between closest points on each rib
    es1 = interpolate_between_closest_two_points_on_rib(s1, 'es')
    es2 = interpolate_between_closest_two_points_on_rib(s2, 'es')
    hs1 = interpolate_between_closest_two_points_on_rib(s1, 'hs')
    hs1 = interpolate_between_closest_two_points_on_rib(s2, 'hs')
    et1 = interpolate_between_closest_two_points_on_rib(t1, 'es')
    et2 = interpolate_between_closest_two_points_on_rib(t2, 'es')
    ht1 = interpolate_between_closest_two_points_on_rib(t1, 'hs')
    ht1 = interpolate_between_closest_two_points_on_rib(t2, 'hs')
    logging.debug(f"Interpolation between points on ribs: es1={es1}, es2={es2}, hs1={hs1}, hs2={hs2}, et1={et1}, et2={et2}, ht1={ht1}, ht2={ht2}") 
    # interpolate elevation and heading between the opposite ribs
    es = (es1 * s2['tmp']['d'] + es2 * s1['tmp']['d']) / (s2['tmp']['d'] + s1['tmp']['d'])
    hs = (hs1 * s2['tmp']['d'] + hs2 * s1['tmp']['d']) / (s2['tmp']['d'] + s1['tmp']['d'])
    et = (et1 * s2['tmp']['d'] + et2 * s1['tmp']['d']) / (s2['tmp']['d'] + s1['tmp']['d'])
    ht = (ht1 * s2['tmp']['d'] + ht2 * s1['tmp']['d']) / (s2['tmp']['d'] + s1['tmp']['d'])
    logging.debug(f"Interpolation points between opposite ribs: es={es}, hs={hs}, et={et}, ht={ht}") 
    # average elevation and heading interpolations
    e = (es + et) / 2
    h = (hs + ht) / 2
    logging.debug(f"Averaged: e={e}, h={h}")
    return e, h

def interpolate_between_closest_two_points_on_rib(rib, et):
    # et is either 'es' for elevation or 'hs' for heading
    # takes weighed average based on the distances to the projection of the input point
    # on the line between the two closest points on the rib
    r = rib['tmp']
    et = (r['d2'] * cal[et][r['i1']] + r['d1'] * cal[et][r['i2']]) / (r['d1'] + r['d2'])
    return et


def read_elevation_and_heading():
    global elevation  # function
    global heading  # function
    global magnetometer, accelerometer
    m = magnetometer.read_average(80)  # averaged sample in uT as np.array
    a = accelerometer.read_ms2()  # sample as np.array
    v = np.hstack((m,a))
    #e = elevation(m[0], m[1], m[2], a[0], a[1], a[2])
    #h = heading(m[0], m[1], m[2], a[0], a[1], a[2])
    #e = elevation(ref_col(v)[0], ref_col(v)[1], ref_col(v)[2])
    #h = heading(ref_col(v)[0], ref_col(v)[1], ref_col(v)[2])
    e, h = rectangular_interpolator(v)
    return v,e,h


def move_and_report():
    global reporting

    def count(start):
        global reporting
        while reporting:
            v, e, h = read_elevation_and_heading()
            #print("elevation: {:4.1f} - heading: {:4.1f}".format(np.degrees(e), np.degrees(h)), end="\r")
            print("x: {:4.1f} - y: {:4.1f} - z: {:4.1f} - elevation: {:4.1f} - heading: {:4.1f}".format(ref_col(v)[0], ref_col(v)[1], ref_col(v)[2], np.degrees(e), np.degrees(h)))
            time.sleep(0.1)

    def start_moving(direction):
        if direction == 'l':
            print("start moving right")
            MOTOR_S.backward()
        elif direction == 'k':
            print("start moving up")
            MOTOR_T.backward()
        elif direction == 'j':
            print("start moving down")
            MOTOR_T.forward()
        elif direction == 'h':
            print("start moving left")
            MOTOR_S.forward()

    reporting = True
    c = threading.Thread(target=count, args=(1,))
    c.start()

    moving = False
    direction = 'k'  # default direction up
    print("Press hjkl for direction and <enter> for stop/restart: ")
    while True:
        char = input("? ")
        if not char and moving:
            print("stop motors")
            MOTOR_S.stop()
            MOTOR_T.stop()
            moving = False
        elif not char and not moving:
            # continue previous direction
            moving = True
            start_moving(direction)
        elif char[0] in "hjkl":
            if moving and not char[0] == direction:
                print("stop motors")
                MOTOR_S.stop()
                MOTOR_T.stop()
            direction = char[0]
            moving = True
            start_moving(direction)
        elif char[0] == 'q':
            break

    reporting = False
    c.join()


def projection_on_vector(v1, v2):
    ''' Returns vector projection of v1 on v2
    '''
    return (np.dot(v1, v2) / np.dot(v2, v2)) * np.array(v2)


def projection_on_plane(v1, n):
    ''' Returns project of v1 on the plane defined by normal vector n
    '''
    return v1 - projection_on_vector(v1, n)


def angle_between_vectors(v1, v2):
    ''' Returns the angle between v1 and v2 in radians
    '''
    v1_unit = v1 / np.linalg.norm(v1)
    v2_unit = v2 / np.linalg.norm(v2)
    cos = np.dot(v1_unit, v2_unit)  # in edge cases this can be like 1.00000000002
    if cos > 1: 
        logging.info(f"Dot product or normalized vectors gave invalid cos of {cos}, making it 1")
        cos = 1
    if cos < -1: 
        logging.info(f"Dot product or normalized vectors gave invalid cos of {cos}, making it -1")
        cos = -1
    angle = np.arccos(cos)
    return angle


def angle_between_vector_and_plane(v, n):
    '''Returns the angle between v and the plane defined by it's normal vector n
    '''
    vp = projection_on_plane(v, n)
    return angle_between_vectors(v, vp)

def point_projected_on_line(a, b, c):
    '''Returns the project of a on the line through b and c
    '''
    bc = np.linalg.norm(c - b)
    d = (c - b) / bc
    v = a - b
    t = np.dot(v, d)
    p = b + t * d
    return p
    
def distance_point_to_line_and_projection_to_end_points(a, b, c):
    '''Returns the distance between a and the line through b and c
       and the distance from the projection of a on this line to resp. b and c 
    '''
    bc = np.linalg.norm(c - b)
    d = (c - b) / bc
    v = a - b
    t = np.dot(v, d)
    p = point_projected_on_line(a, b, c)
    return np.linalg.norm(p - a), t, abs(bc-t)

def point_inbetween_two_other_points(p, a, b):
    '''Returns true if p is inbetween a and b (all points supposed to be colinear
    '''
    ap = p - a
    bp = p - b
    # only if p is inbetween a and b, ap and bp have opposite direction
    # so ||ap+bp|| <= ||ap-bp||  (including the edge case where p = a or p = b)
    # if p is not inbetween a and b, ap and bp have the same direction
    # so ||ap+bp|| > ||ap-bp||
    return np.linalg.norm(ap + bp) <= np.linalg.norm(ap - bp)

def point_inbetween_two_other_points_3d(p, a, b):
    '''Returns True if projection of p on line ab is between a and b
    '''
    p2 = point_projected_on_line(p, a, b)
    return point_inbetween_two_other_points(p2, a, b)

def plot(data=None, title=None, final=False):
    global plot_data
    global plot_title
    global plot_final
    plot_data = data
    plot_title = title
    plot_final = final
    if __name__.startswith('bokeh'):
        handler_lock.release()
        generator_lock.acquire()

# THIS RUNS IN A THREAD!
def menu():
    logging.info("Entering menu()")
    options = [
        ['Calibrate', calibrate],
        ['Save calibration to file', save_calibration],
        ['Load calibration from file', load_calibration],
        ['Move using keyboard and report position', move_and_report],
        ['Retract motors', retract_motors],
        ['Quit', sys.exit]
    ]
    while True:
        try:
            for index, option in enumerate(options):
                print(f"{index} - {option[0]}")
            choice = int(input("Choose number: "))
            options[choice][1]()  # run the function stored as second element in the chosen row of options
        except SystemExit:
            sys.exit()  # a bit silly, but the sys.exit() call from the menu 
                        # is otherwise caught by the general exception
        except:
            logging.exception("Procedure failed; returning to menu.")
            continue

# Setup sensor
accelerometer = Accelerometer()
magnetometer = Magnetometer()
gyroscope = Gyroscope()

# start menu in a separate thread
t = threading.Thread(target=menu)
t.start()

# draw the start button
button = Button(label="Start handling data", button_type="success")
button.on_click(data_handler)
curdoc().add_root(button)

# MAIN SCRIPT ENDS HERE, SO BOKEH CAN BOOTSTRAP
