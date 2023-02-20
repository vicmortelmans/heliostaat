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
import datetime
from gpiozero import Motor
import imutils
import json
import logging
import math
import numpy as np
from picamera.array import PiRGBArray
from picamera import PiCamera
from pysolar.solar import get_altitude, get_azimuth
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
MOTOR_TRAVEL_TIME = 25  # 25 seconds, based on experiment, 10 for speeding up durig debugging
MOTOR_S = Motor(26, 20)  # GPIO pins connected to input leads 'forward' and 'backward'
MOTOR_T = Motor(19, 16)
# https://gpiozero.readthedocs.io/en/stable/recipes.html?highlight=motor#motors
# https://gpiozero.readthedocs.io/en/v1.2.0/api_output.html#motor
# L298N motor driver module H-Bridge

# Setup camera resolution, based on camera specs
camera_w = 2592
camera_h = 1944

# Setup camera calibration data
DIM=(2592, 1944)
mtx=np.array([[1513.35202186325, 0.0, 1381.794375023546], [0.0, 1514.809082655238, 1022.1313014429818], [0.0, 0.0, 1.0]])
dist=np.array([[-0.3293226333311312, 0.13030355339675337, 0.00020716954584170977, -0.00032937886446441326, -0.027128518075549755]])
# Output of scripts in camera_calibration directory

# Setup camera field of view calibration data global variable
Lh = 188.3  # visible length Ll of horizontal ruler (cm)
Dh = 82.0  # at distance Dl of camera (cm)
Lv = 145.5  # visible length Ll of vertical ruler (cm)
Dv = 79.5  # at distance Dl of camera (cm)

# Setup location
lat = 51.2109917 
lon = 4.4282524

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
    global accelerometer
    a = accelerometer.read_ms2()
    # level is angle between gravity vector and XZ plane
    # looking at the mirror, level is negative if it's turned to the right (ay < 0)
    level = angle_between_vector_and_plane(a, np.array([0,1,0]))
    if a[1] < 0:
        level = -level
    # helio_tilt is angle between gravity vector and XY plane
    # looking at the mirror, helio_tilt is negative if it's turned forward (az > 0)
    helio_tilt = angle_between_vector_and_plane(a, np.array([0,0,1]))
    if a[2] > 0:
        helio_tilt = -helio_tilt
    return level, helio_tilt

def move_for_seconds(MOTOR=None, motorname=None, duration=0, forward=False):
    logging.info(f"Start moving {motorname} motor")
    if forward:
        MOTOR.forward()
    else:
        MOTOR.backward()
    time.sleep(duration)
    logging.info(f"Stop {motorname} motor")
    MOTOR.stop()


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

def capture_image():
    # constant parameters obtained by running camera calibration
    global DIM, mtx, dist
    global Lh, Dh, Lv, Dv

    global camera, rawCapture
    # capture image
    camera.capture(rawCapture, format="bgr")
    img = rawCapture.array
    logging.debug(f"Captured image with dimensions {img.shape}")
    rawCapture.truncate(0)
    
    dim1 = img.shape[:2][::-1]  #dim1 is the dimension of input image to un-distort
    assert (dim1[0] == DIM[0]) and (dim1[1] == DIM[1]), "Image to undistort needs to have same dimensions as the ones used in calibration"

    # undistort
    h,  w = img.shape[:2]
    newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1.0,(w,h))
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

    # crop the image (increasing the cropping by 20%)
    x,y,w,h = roi
    o = int(y-.2*h)
    b = int(y+h+.2*h)
    l = int(x-.2*w)
    r = int(x+w+.2*w)
    dst = dst[o:b, l:r]
    logging.debug(f"Undistorted image after cropping has dimensions {dst.shape}")

    # stretch the image to fix aspect ratio
    h, w = dst.shape[:2]
    c = Lh * Dv * h / Lv / Dh / w
    logging.debug(f"Restoring aspect ratio by resizing vertically with factor {c}")
    asp = cv2.resize(dst, (w, int(w / c)))
    logging.debug(f"Image after restoring aspect ratio has dimensions {asp.shape}")

    # rotate the image
    level, _ = read_level()
    logging.debug(f"Level: {np.degrees(level)}")
    rot = imutils.rotate_bound(asp, 180 + np.degrees(level))  # 180 because camera is mounted upside down
    # rotate_bound positive angle is clockwise
    logging.debug(f"Image after rotation has dimensions {rot.shape}")

    return rot


def read_sun_to_mirror():

    # capture image
    image_straight = capture_image()

    # find brightest point pixel position
    gray = cv2.cvtColor(image_straight, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (17,17), 0)
    (_, _, _, (x,y)) = cv2.minMaxLoc(gray)
    logging.debug(f"Brightest spot at ({x}, {y})")

    cv2.circle(image_straight, (x,y), 10, (255, 0, 0), 2)
    cv2.imshow("sun", image_straight)
    cv2.waitKey(1000)
    #cv2.destroyAllWindows()
    #cv2.imwrite('image.png',rot)

    # convert to pixel position relative to center lines
    (w, h) = gray.shape[:2][::-1]
    xn = x - w/2 
    yn = - y + h/2 
    logging.debug(f"Brightest spot pixel position relative to center ({xn}, {yn})")

    # calculate horizontal (swipe - heading) and vertikal (tilt - elevation) angles
    efh = math.atan(Lh * xn / Dh / w) 
    efv = math.atan(Lh * yn / Dh / w) 
    logging.debug(f"Brightest spot heading and elevation ({np.degrees(efh)}, {np.degrees(efv)})")

    return efh, efv


def read_target_to_horizon():
    date = datetime.datetime.now(datetime.timezone.utc)
    eh = get_azimuth(lat, lon, date)
    ev = get_altitude(lat, lon, date)
    logging.info(f"Current position of the sun relative to horizon: altitude {ev}, azimuth {eh}")
    efh, efv = read_sun_to_mirror()
    logging.info(f"Current position of the sun relative to mirror: altitude {efv}, azimuth {efh}")
    eoh = 2 * efh - eh
    eov = 2 * efv - ev
    logging.info(f"Position of the target relative to horizon: altitude {eov}, azimuth {eoh}")
    return eoh, eov


def move_mirror_to_relative_position_to_sun_interactively():
    th = np.radians(float(input("Horizontal angle of view: ")))
    tv = np.radians(float(input("Vertical angle of view: ")))
    move_to_target(th, tv)


def move_to_target(th, tv):
    global Lh, Dh, Lv, Dv
    ACCURACY = np.radians(1)  # 1 degree
    fovh = math.atan(Lh/2/Dh)  # maximum horizontal viewing angle (= 50% of total field of view)
    fovv = math.atan(Lv/2/Dv)  # maximum vertical viewing angle (= 50% of total field of view)
    offsh = fovh
    offsv = fovv
    stepT = MOTOR_TRAVEL_TIME / 4  # somewhat less than 50% of total span
    stepS = MOTOR_TRAVEL_TIME / 4
    efh, efv = read_sun_to_mirror()  # viewing angles
    noffsh = th - efh
    noffsv = tv - efv
    logging.info(f"Current position of the sun in the photo: {np.degrees((efh, efv))}; target: {np.degrees((th, tv))}")
    while abs(noffsh) > ACCURACY or abs(noffsv) > ACCURACY:
        if abs(noffsh) > ACCURACY:
            stepS = abs(stepS * noffsh / offsh) if offsh else stepS
            motorname = "swivel"
            direction = False if noffsh > 0 else True  # 'moving' the sun down in the photo means collapsing the mirror
            # NOTE: the directions are flipped if the mirror hinge would be mounted on the other side!
            logging.debug(f"Angle {motorname} {'extending' if direction else 'collapsing'} {stepS} seconds towards {np.degrees(noffsh)} degrees offset")
            move_for_seconds(MOTOR=MOTOR_S, motorname=motorname, duration=stepS, forward=direction)
            efh, efv = read_sun_to_mirror()
            offsh = noffsh
            noffsh = th - efh
            if abs(noffsh) > ACCURACY and abs(noffsh / offsh) > 0.9:
                logging.warning(f"Gone beyond swivel movement range: offsh {offsh} => noffsh {noffsh}")
                break
        if abs(noffsv) > ACCURACY:
            noffsv = tv - efv
            stepT = abs(stepT * noffsv / offsv) if offsv else stepT
            motorname = "tilt"
            direction = True if noffsv > 0 else False  # 'moving' the sun up in the photo means extending the mirror
            logging.debug(f"Angle {motorname} {'extending' if direction else 'collapsing'} {stepT} seconds towards {np.degrees(noffsv)} degrees offset")
            move_for_seconds(MOTOR=MOTOR_T, motorname=motorname, duration=stepT, forward=direction)
            efh, efv = read_sun_to_mirror()  # viewing angles
            offsv = noffsv
            noffsv = tv - efv
            if abs(noffsv) > ACCURACY and abs(noffsv / offsv) > 0.9:
                logging.warning(f"Gone beyond tilt movement range: offsv {offsv} => noffsv {noffsv}")
                break
            noffsh = th - efh
        logging.debug(f"Offset at end of loop: {np.degrees((noffsh, noffsv))}")


def move_and_report():
    global reporting

    def count(start):
        global reporting
        while reporting:
            efh, efv = read_sun_to_mirror()
            print("sun relative to mirror elevation: {:.3f} - heading: {:.3f}".format(np.degrees(efv), np.degrees(efh)))
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
        ['Read target position relative to horizon', read_target_to_horizon],
        ['Move using keyboard and report position', move_and_report],
        ['Move to relative position to sun interactively', move_mirror_to_relative_position_to_sun_interactively],
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

# setup camera
camera = PiCamera()
camera.resolution = (camera_w, camera_h)
rawCapture = PiRGBArray(camera)

# demo sun position
date = datetime.datetime.now(datetime.timezone.utc)
logging.info(f"Current position of the sun: altitude {get_altitude(lat, lon, date)}, azimuth {get_azimuth(lat, lon, date)}")

# start menu in a separate thread
t = threading.Thread(target=menu)
t.start()

# draw the start button
button = Button(label="Start handling data", button_type="success")
button.on_click(data_handler)
curdoc().add_root(button)

# MAIN SCRIPT ENDS HERE, SO BOKEH CAN BOOTSTRAP
