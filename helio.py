#!/usr/bin/python3

# Heliostat driver - this script should be run as a service

# In the end, this script should provide the remote IoT interface

# During development, it's invoked on command line

# Run with bokeh:
#   bokeh serve --allow-websocket-origin=nymea.local:5006 helio.py
# Then open the website to continue the script

import adafruit_fxos8700
import adafruit_fxas21002c
import board
from bokeh.plotting import curdoc, figure
from bokeh.models import Button
import cv2
from gpiozero import Motor
import json
import logging
import numpy as np
from picamera.array import PiRGBArray
from picamera import PiCamera
from scipy import linalg, signal, interpolate
import scipy.ndimage
import sys
import threading
import time
import vg

handler_lock = threading.Semaphore(0)  # handler waiting for data
generator_lock = threading.Semaphore(0)  # generator waiting for user pushing the button

# Setup logging
logging.basicConfig(force=True, format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d %(funcName)s] %(message)s', datefmt='%Y-%m-%d:%H:%M:%S', level=logging.DEBUG)
# force=True is needed because bokeh sets the level down

# Setup motors
MOTOR_TRAVEL_TIME = 25  # 25 seconds, 10 for speeding up durig debugging
MOTOR_S = Motor(26, 20)
MOTOR_T = Motor(19, 16)

# Setup calibration data global variable
Lh = 0.0  # visible length Ll of horizontal ruler
Dh = 0.0  # at distance Dl of camera
Lv = 0.0  # visible length Ll of vertical ruler
Dv = 0.0  # at distance Dl of camera

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



def retract(MOTOR=None, motorname=None):
    logging.info(f"Start fully retract {motorname} motor")
    MOTOR.backward()
    time.sleep(MOTOR_TRAVEL_TIME)
    logging.info(f"Stop fully retract {motorname} motor")
    MOTOR.stop()

def retract_motors():
    retract(MOTOR=MOTOR_S, motorname="sweep")
    retract(MOTOR=MOTOR_T, motorname="tilt")


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

def undistort(img, balance=0.0, dim2=None, dim3=None):
    # constant parameters obtained by running camera calibration
    DIM=(2592, 1944)
    mtx=np.array([[1513.35202186325, 0.0, 1381.794375023546], [0.0, 1514.809082655238, 1022.1313014429818], [0.0, 0.0, 1.0]])
    dist=np.array([[-0.3293226333311312, 0.13030355339675337, 0.00020716954584170977, -0.00032937886446441326, -0.027128518075549755]])

    dim1 = img.shape[:2][::-1]  #dim1 is the dimension of input image to un-distort

    assert (dim1[0] == DIM[0]) and (dim1[1] == DIM[1]), "Image to undistort needs to have same dimensions as the ones used in calibration"

    if not dim2:
        dim2 = dim1

    if not dim3:
        dim3 = dim1

    h,  w = img.shape[:2]
    newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))

    # undistort
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

    # crop the image
    x,y,w,h = roi
    dst = dst[y:y+h, x:x+w]
    return dst


def read_sun_to_mirror():
    global camera, rawCapture
    # capture image
    camera.capture(rawCapture, format="bgr")
    image = rawCapture.array

    # undistort image
    image_straight = undistort(image, balance=0.8)

    # find brightest point pixel position
    gray = cv2.cvtColor(image_straight, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (32,32), 0)
    (_, _, _, (x,y)) = cv2.minMaxLoc(gray)

    # convert to normalized pixel position relative to center lines
    (w, h) = gray.shape[:2][::-1]
    xn = 2 * x / w - 1
    yn = - 2 * y / h + 1 

    # 
    efh = atan(Lh * x / 2 / Dh) 


def move_and_report():
    global reporting

    def count(start):
        global reporting
        while reporting:
            v, e, h = read_elevation_and_heading()
            #print("elevation: {:4.1f} - heading: {:4.1f}".format(np.degrees(e), np.degrees(h)), end="\r")
            print("x: {:.3f} - y: {:.3f} - z: {:.3f} - elevation: {:.3f} - heading: {:.3f}".format(ref_col(v)[0], ref_col(v)[1], ref_col(v)[2], np.degrees(e), np.degrees(h)))
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
        ['Read sun position relative to mirror', read_sun_to_mirror],
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

# setup camera
camera = PiCamera()
rawCapture = PiRGBArray(camera)

# start menu in a separate thread
t = threading.Thread(target=menu)
t.start()

# draw the start button
button = Button(label="Start handling data", button_type="success")
button.on_click(data_handler)
curdoc().add_root(button)

# MAIN SCRIPT ENDS HERE, SO BOKEH CAN BOOTSTRAP
