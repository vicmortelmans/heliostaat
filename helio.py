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
from scipy.interpolate import LinearNDInterpolator
import threading
import time
import vg

handler_lock = threading.Semaphore(0)  # handler waiting for data
generator_lock = threading.Semaphore(0)  # generator waiting for user pushing the button

# Setup logging
logging.basicConfig(force=True, format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d %(funcName)s] %(message)s', datefmt='%Y-%m-%d:%H:%M:%S', level=logging.INFO)
# force=True is needed because bokeh sets the level down

# Setup motors
MOTOR_TRAVEL_TIME = 30  # 30 seconds, 10 for speeding up durig debugging
MOTOR_S = Motor(26, 20)
MOTOR_T = Motor(19, 16)

# Setup calibration data global variable
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

    def read_raw(self):
        ''' Get a sample.

            Returns
            -------
            s : np.array
                The reading in uT (not corrected).
        '''
        return np.array(self.sensor.read())


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
    # starting from the end, find the first value where change > threshold
    i = len(xs) - 1
    x = xs[i]
    while abs(x - xs[i]) < threshold:
        i = i-1
        if i == 0:
            logging.error(f"No tail found")
            return 0
    logging.info(f"Tail found at value {i} of {len(xs)}")
    return i + 1

def trim_tail(vs):
    # figure out when the values stabilize and 
    # return the index and the trimmed and smoothened vectors
    m_threshold = 2.0
    a_threshold = 0.2
    # reduce noisiness 
    vs = np.array(vs)
    l = len(vs)
    # smoothen the readings
    mxs = signal.savgol_filter(vs[:,0], 481, 2)
    mys = signal.savgol_filter(vs[:,1], 481, 2)
    mzs = signal.savgol_filter(vs[:,2], 481, 2)
    axs = signal.savgol_filter(vs[:,3], 481, 2)
    ays = signal.savgol_filter(vs[:,4], 481, 2)
    azs = signal.savgol_filter(vs[:,5], 481, 2)
    logging.info("Finding tails for resp. mx, my, mz, ax, ay, az")
    plot(data={'x': np.linspace(1, len(mxs), len(mxs)), 'y': mxs}, title="mxs")
    plot(data={'x': np.linspace(1, len(mys), len(mys)), 'y': mys}, title="mys")
    plot(data={'x': np.linspace(1, len(mzs), len(mzs)), 'y': mzs}, title="mzs")
    plot(data={'x': np.linspace(1, len(axs), len(axs)), 'y': axs}, title="axs")
    plot(data={'x': np.linspace(1, len(ays), len(ays)), 'y': ays}, title="ays")
    plot(data={'x': np.linspace(1, len(azs), len(azs)), 'y': azs}, title="azs")
    tails = []
    tails.append(find_tail(mxs, m_threshold))
    tails.append(find_tail(mys, m_threshold))
    tails.append(find_tail(mzs, m_threshold))
    tails.append(find_tail(axs, a_threshold))
    tails.append(find_tail(ays, a_threshold))
    tails.append(find_tail(azs, a_threshold))
    # for sanity, check if tails > 1/2
    tails = np.extract(np.greater(tails, l/2), tails)
    logging.info(f"From 6 tails, {len(tails)} were valid")
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
    off = np.sqrt((1/np.tan(ang/2)+q1)**2 + (1/np.tan(ang/2)+q2)**2 - 2*(1/np.tan(ang/2)+q1)*(1/np.tan(ang/2)+q2)*np.cos(ang))
    ang = np.hstack((0,ang))
    off = np.hstack((0,off))
    return interpolate.interp1d(off, ang)

def process_swipe(times, vs, forward=True):
    # trim the readouts to where the movement stops (= 90 degrees)
    # and convert times to angles
    normalized_offset_to_hinge_angle = normalized_offset_to_hinge_angle_function()
    timeindexmax, vs = trim_tail(vs) 
    times = times[0:timeindexmax]
    timemax = times[-1]  # last element = times[timeindexmax-1]
    if not forward:
        # swiping backward, so flipping order of samples, because normalizing is direction-dependent!
        times = np.flip(times)
        vs = np.flip(vs, 0)
    angs = normalized_offset_to_hinge_angle(times/timemax)
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

def register(ss, ts, vs):
    global cal
    if not cal:
        cal['ss'] = ss
        cal['ts'] = ts
        cal['vs'] = vs
    else:
        cal['ss'] = np.hstack((cal['ss'], ss))
        cal['ts'] = np.hstack((cal['ts'], ts))
        cal['vs'] = np.vstack((cal['vs'], vs))

def register_array(array, name):
    global cal
    cal[name] = array

def normal_vectors(helio_tilt):
    global cal
    ht = helio_tilt  # readibility
    for i in range(len(cal['vs'])):
        s = cal['ss'][i]
        t = cal['ts'][i]
        x = -np.sin(s)*np.cos(ht)
        y = np.cos(t)*np.cos(s)*np.cos(ht) - np.sin(t)*np.sin(ht)
        z = np.sin(t)*np.sin(s)*np.cos(ht) + np.cos(t)*np.sin(ht)
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
    # wait until data is ready
    print("Handler: waiting for data from generator")
    handler_lock.acquire()
    print("Handler: handling data from generator")
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
    # unlock the generator to continue generating data
    print("Handler: unlocking generator")
    generator_lock.release()
    # actual drawing is only done after the callback finishes!

def calibrate():
    global cal
    global interpol
    global elevation  # function
    global heading  # function

    cal = {}  # reset global variable containing the calibration values

    if NUMBER_OF_SCANS > 1:
        partial_motor_travel_time = MOTOR_TRAVEL_TIME / (NUMBER_OF_SCANS - 1)
    else:
        partial_motor_travel_time = MOTOR_TRAVEL_TIME

    logging.info("Start heliostat position calibration")
    retract(MOTOR=MOTOR_T, motorname="tilt")
    retract(MOTOR=MOTOR_S, motorname="swivel")

    # calculate the tilt angle of the whole heliostat
    level, helio_tilt = read_level()
    if level > np.pi/20:
        logging.warning(f"The heliostat isn't mounted level, it's {level} rad off.")
    logging.info(f"The heliostat is mounted with a tilt of {helio_tilt} rad.")

    # sequence of forward partial swivel swipes with 0 tilt angle 

    for i in range(NUMBER_OF_SCANS):
        partial_times, partial_vs = partial_swipe(duration=partial_motor_travel_time, forward=True, MOTOR=MOTOR_S, motorname="swivel")
        if i == 0:
            times = partial_times
            vs = partial_vs
            swipe_idxs = [len(times) - 1]  # list of indices of end times for each partial swipe
        else:
            times = np.hstack((times, partial_times + swipe_times[-1]))  # starting from end time in previous partial swipe
            vs = np.vstack((vs, partial_vs))
            swipe_idxs.append(swipe_idxs[-1] + len(times))  
        ss, vs = process_swipe(times, vs, forward=True)
        register(ss, np.full((1, len(ss)), 0), vs)  # ts are all zero
        # make a list of the swivel angles for constant swivel angle
        s_angles = [ss[i] for i in swipe_idxs if i < len(ss)]  # the number angles may be lower than NUMBER_OF_SCANS

    # sequence of forward partial tilt swipes with 90 swivel angle 

    for i in range(NUMBER_OF_SCANS):
        partial_times, partial_vs = partial_swipe(duration=partial_motor_travel_time, forward=True, MOTOR=MOTOR_T, motorname="tilt")
        if i == 0:
            times = partial_times
            vs = partial_vs
            swipe_idxs = [len(times) - 1]  # list of indices of end times for each partial swipe
        else:
            times = np.hstack((times, partial_times + swipe_times[-1]))  # starting from end time in previous partial swipe
            vs = np.vstack((vs, partial_vs))
            swipe_idxs.append(swipe_idxs[-1] + len(times))  
        ts, vs = process_swipe(times, vs, forward=True)
        register(np.full((1, len(ts)), np.pi/2), ts, vs)  # ss are all 90 degrees
        # make a list of the tilt angles for constant tilt angle
        t_angles = [ts[i] for i in swipe_idxs if i < len(ss)]  # the number angles may be lower than NUMBER_OF_SCANS

    # sequence of backward partial swivel swipes with 90 tilt angle 

    for i in range(NUMBER_OF_SCANS):
        partial_times, partial_vs = partial_swipe(duration=partial_motor_travel_time, forward=False, MOTOR=MOTOR_S, motorname="swivel")
        if i == 0:
            times = partial_times
            vs = partial_vs
        else:
            times = np.hstack((times, partial_times + swipe_times[-1]))  # starting from end time in previous partial swipe
            vs = np.vstack((vs, partial_vs))
        ss, vs = process_swipe(times, vs, forward=False)
        register(ss, np.full((1, len(ss)), np.pi/2), vs)  # ts are all 90 degrees

    # sequence of backward partial tilt swipes with 0 swivel angle 

    for i in range(NUMBER_OF_SCANS):
        partial_times, partial_vs = partial_swipe(duration=partial_motor_travel_time, forward=False, MOTOR=MOTOR_T, motorname="tilt")
        if i == 0:
            times = partial_times
            vs = partial_vs
        else:
            times = np.hstack((times, partial_times + swipe_times[-1]))  # starting from end time in previous partial swipe
            vs = np.vstack((vs, partial_vs))
        ts, vs = process_swipe(times, vs, forward=True)
        register(np.full((1, len(ts)), 0), ts, vs)  # ss are all 0

    # swipes with constant tilt angle

    forward = True
    for t in t_angles:  # skipping the first scan at 0 degrees
        partial_swipe(duration=partial_motor_travel_time, forward=True, MOTOR=MOTOR_T, motorname="tilt")
        times, vs = swipe(forward=forward, MOTOR=MOTOR_S, motorname="swivel")
        ss, vs = process_swipe(times, vs, forward=forward)
        register(ss, np.full((1, len(ss)), t), vs)  # ts are all t (from t_angles)
        forward = not forward  # reverse direction for next iteration
    if not forward:  # last swipe was forward
        retract(MOTOR=MOTOR_S, motorname="swivel")
    retract(MOTOR=MOTOR_T, motorname="tilt")

    # swipes with constant swivel angle

    forward = True
    for s in s_angles:
        partial_swipe(duration=partial_motor_travel_time, forward=True, MOTOR=MOTOR_S, motorname="swivel")
        times, vs = swipe(forward=forward, MOTOR=MOTOR_T, motorname="tilt")
        ts, vs = process_swipe(times, vs, forward=forward)
        register(np.full((1, len(ts)), s), ts, vs)  # ss are all s (from s_angles)
        forward = not forward  # reverse direction for next iteration
    if not forward:  # last swipe was forward
        retract(MOTOR=MOTOR_T, motorname="tilt")
    retract(MOTOR=MOTOR_S, motorname="swivel")

    # calculate for each sample the absolute normal vector of the mirror
    # (in coordinate system with vertical Z, thus including the effect of the helio tilt)
    register_by_name(normal_vectors(helio_tilt), 'ns')
    # calculate for each sample the mirror elevation and heading
    register_by_name(elevations(), 'es')
    register_by_name(headings(), 'hs')
    # initialize interpolcation functions
    elevation = LinearNDInterpolator(cal['vs'], cal['es'])
    heading = LinearNDInterpolator(cal['vs'], cal['hs'])

    logging.info("Finish magnetometer calibration")
    logging.info("Finished heliostat position calibration")


def save_calibration():
    global cal
    logging.info("Writing data to file")
    np.savetext('ss.csv', cal['ss'])
    np.savetext('ts.csv', cal['ts'])
    np.savetext('vs.csv', cal['vs'])
    np.savetext('ns.csv', cal['ns'])
    np.savetext('es.csv', cal['es'])
    np.savetext('hs.csv', cal['hs'])


def load_calibration():
    global cal
    global elevation  # function
    global heading  # function
    cal['ss'] = np.loadtext('ss.csv')
    cal['ts'] = np.loadtext('ts.csv')
    cal['vs'] = np.loadtext('vs.csv')
    cal['ns'] = np.loadtext('ns.csv')
    cal['es'] = np.loadtext('es.csv')
    cal['hs'] = np.loadtext('hs.csv')
    # initialize interpolcation functions
    elevation = LinearNDInterpolator(cal['vs'], cal['es'])
    heading = LinearNDInterpolator(cal['vs'], cal['hs'])


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

def plot(data=None, title=None, final=False):
    global plot_data
    global plot_title
    global plot_final
    plot_data = data
    plot_title = title
    plot_final = final
    handler_lock.release()
    generator_lock.acquire()

# THIS RUNS IN A THREAD!
def menu():
    logging.info("Entering menu()")
    options = [
        ['Calibrate', calibrate],
        ['Save calibration to file', save_calibration],
        ['Load calibration from file', load_calibration],
        ['Track inclination and heading angles', track_inclination_and_heading],
        ['Track magnetic sensor', track_magnetic],
        ['Retract motors', retract_motors]
    ]
    while True:
        try:
            for index, option in enumerate(options):
                print(f"{index} - {option[0]}")
            choice = int(input("Choose number: "))
            options[choice][1]()  # run the function stored as second element in the chosen row of options
        except:
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
