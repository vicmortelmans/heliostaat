#!/usr/bin/python3

# Heliostat driver - this script should be run as a service

# In the end, this script should provide the remote IoT interface

# During development, it's invoked on command line

import adafruit_fxos8700
import adafruit_fxas21002c
from ahrs import Quaternion
from ahrs.filters import Madgwick, Mahony
import board
import csv
from gpiozero import Motor
import logging
import numpy as np
import random
from scipy import linalg, signal
from simple_term_menu import TerminalMenu
import threading
import time
import vg

# Setup logging
logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d %(funcName)s] %(message)s', datefmt='%Y-%m-%d:%H:%M:%S', level=logging.DEBUG)

# Setup motors
MOTOR_TRAVEL_TIME = 10  # 30 seconds, 10 for speeding up durig debugging
MOTORH = Motor(26, 20)
MOTORA = Motor(19, 16)

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
        self._calibration_s = []  # list of raw sensor readings during calibration
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

    def launch_calibration(self):
        ''' Performs calibration, actually only the loop collecting the data,
            running in a separate thread. It's monitoring the _calibration_complete
            property, set by the finish_calibration method, to stop. 
            The caller must take care of triggering the motion of the sensor and
            of calling finish_calibraiton when done.
        '''

        if self._calibration_complete == True:
            logging.error("Trying to launch calibration before previous one finished!")
            return
        self._calibration_s = []
        self._calibration_complete = False
        threading.Thread(target=self.__collect_calibration_data).start()


    def __collect_calibration_data(self):
        logging.info("Starting to collect samples for magnetometer calibration in thread")
        while self._calibration_complete == False:
            m = self.read_raw()  # sample in uT
            self._calibration_s.append(m)
            print("x,y,z: {:8.1f},{:8.1f},{:8.1f}".format(m[0], m[1], m[2]), end="\r")
        logging.info("Stopped collecting samples for magnetometer calibration in thread")


    def finish_calibration(self):
        ''' Finishes the calibration, by setting the calibration_complete property
            to finish data collection by the launch_calibration thread and then
            calculating the calibration result
        '''

        # finish reading sensor data
        self._calibration_complete = True
        s = self._calibration_s
        count = len(self._calibration_s)
        logging.info(f"Collected {count} samples for magnetometer calibration")

        # average strength
        strength = 0
        for v in s:
            strength += np.linalg.norm(v) / count
        logging.info(f"Average strength of the samples is {strength} nT")

        # ellipsoid fit
        s = np.array(s).T
        M, n, d = self.__ellipsoid_fit(s)

        # calibration parameters
        # note: some implementations of sqrtm return complex type, taking real
        M_1 = linalg.inv(M)
        self.b = -np.dot(M_1, n)
        self.A_1 = np.real(49.081 / np.sqrt(np.dot(n.T, np.dot(M_1, n)) - d) *
                           linalg.sqrtm(M))

        # clear way for next calibration
        self._calibration_complete = None


    def save_calibration(self):
        ''' Saves calibration data to two files magcal_b.npy and magcal_A_1.npy
            in current directory
        '''

        logging.debug("Magnetic calibration going to be saved to file")
        np.save('magcal_b.npy', self.b)
        np.save('magcal_A_1.npy', self.A_1)
        logging.info("Magnetic calibration saved to file")


    def load_calibration(self):
        ''' Loads calibration data from two files magcal_b.npy and magcal_A_1.npy
            in current directory
        '''

        logging.debug("Magnetic calibration going to be loaded from file")
        self.b = np.load('magcal_b.npy')
        self.A_1 = np.load('magcal_A_1.npy')
        logging.info("Magnetic calibration loaded from file")


    def __ellipsoid_fit(self, s):
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



def calibrate_magnetometer():
    global magnetometer
    magnetometer.launch_calibration()  # starts reading
    logging.info("Start magnetometer calibration")
    logging.info("Start fully retract azimut motor")
    MOTORA.backward()
    time.sleep(MOTOR_TRAVEL_TIME)
    logging.info("Stop fully retract azimut motor")
    MOTORA.stop()
    logging.info("Start fully retract heading motor")
    MOTORH.backward()
    time.sleep(MOTOR_TRAVEL_TIME)
    logging.info("Stop fully retract heading motor")
    MOTORH.stop()
    motorh_due_forward = True
    remaining_amotor_travel_time = MOTOR_TRAVEL_TIME
    if NUMBER_OF_SCANS > 1:
        partial_amotor_travel_time = MOTOR_TRAVEL_TIME / (NUMBER_OF_SCANS - 1)
    while remaining_amotor_travel_time >= 0:
        if motorh_due_forward:
            logging.info("Start fully extend heading motor")
            MOTORH.forward()
            time.sleep(MOTOR_TRAVEL_TIME)
            logging.info("Stop fully extend heading motor")
            MOTORH.stop()
        else:
            logging.info("Start fully retract heading motor")
            MOTORH.backward()
            time.sleep(MOTOR_TRAVEL_TIME)
            logging.info("Stop fully retract heading motor")
            MOTORH.stop()
        if NUMBER_OF_SCANS == 1:
            break
        logging.info("Start partially extend azimut motor")
        MOTORA.forward()
        time.sleep(partial_amotor_travel_time)
        MOTORA.stop()
        logging.info("Stop partially extend azimut motor")
        remaining_amotor_travel_time -= partial_amotor_travel_time
    logging.info("Start fully retract azimut motor")
    MOTORA.backward()
    time.sleep(MOTOR_TRAVEL_TIME)
    logging.info("Stop fully retract azimut motor")
    logging.info("Finish magnetometer calibration")
    magnetometer.finish_calibration()  # stops reading
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


def calibrate():
    global cal
    logging.info("Start calibration")
    logging.info("Start fully retract azimut motor")
    MOTORA.backward()
    time.sleep(MOTOR_TRAVEL_TIME)
    logging.info("Stop fully retract azimut motor")
    MOTORA.stop()
    cal = { 'acxs': [], 'acys': [], 'aczs': [], 'tcxss': [], 'tcyss': [], 'tczss': [] }
    remaining_amotor_travel_time = MOTOR_TRAVEL_TIME
    if NUMBER_OF_SCANS > 1:
        partial_amotor_travel_time = MOTOR_TRAVEL_TIME / (NUMBER_OF_SCANS - 1)
    while remaining_amotor_travel_time >= 0:
        logging.info("Start fully retract heading motor")
        MOTORH.backward()
        time.sleep(MOTOR_TRAVEL_TIME)
        logging.info("Stop fully retract heading motor")
        MOTORH.stop()
        acx, acy, acz = sensor.accelerometer
        tcxs, tcys, tczs = scan()
        cal['acxs'].append(acx)
        cal['acys'].append(acy)
        cal['aczs'].append(acz)
        cal['tcxss'].append(signal.savgol_filter(tcxs, 9, 1))
        cal['tcyss'].append(signal.savgol_filter(tcys, 9, 1))
        cal['tczss'].append(signal.savgol_filter(tczs, 9, 1))
        if NUMBER_OF_SCANS == 1:
            break
        logging.info("Start partially extend azimut motor")
        MOTORA.forward()
        time.sleep(partial_amotor_travel_time)
        MOTORA.stop()
        logging.info("Stop partially extend azimut motor")
        remaining_amotor_travel_time -= partial_amotor_travel_time
    logging.info("Start fully retract azimut motor")
    MOTORA.backward()
    time.sleep(MOTOR_TRAVEL_TIME)
    logging.info("Stop fully retract azimut motor")
    logging.info("Stop calibration")


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
            q_now = madgwick.updateMARG(q_previous, gyr=g, acc=a, mag=1000*m, dt=time_delay)
            time_previous = time_now
            q_previous = q_now
            e = Quaternion(q_now).to_angles()
            print("roll: {:4.1f} - pitch: {:4.1f} - yaw: {:4.1f} - frequency: {:4.1f}".format(np.degrees(e[0]), np.degrees(e[1]), np.degrees(e[2]), 1/time_delay), end="\r")
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
            q_now = mahony.updateMARG(q_previous, gyr=g, acc=a, mag=1000*m, dt=time_delay)
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
            print("azimut: {:4.1f} - heading: {:4.1f}".format(aangle, tangle), end="\r")
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass



def scan():
    logging.info("Start heading scan")
    tcxs = []
    tcys = []
    tczs = []
    stop = time.time() + MOTOR_TRAVEL_TIME  # seconds
    logging.info("Start full travel extending of heading motor")
    MOTORH.forward()
    while time.time() < stop:
        tcx, tcy, tcz = sensor.magnetometer
        tcxs.append(tcx)
        tcys.append(tcy)
        tczs.append(tcz)
        time.sleep(0.1)
    logging.info("Stop full travel extending of heading motor")
    MOTORH.stop()
    logging.info("Stop heading scan")
    return tcxs, tcys, tczs


def main():
    global accelerometer, magnetometer, gyroscope
    accelerometer = Accelerometer()
    magnetometer = Magnetometer()
    gyroscope = Gyroscope()
    options = [
        ['Calibrate magnetometer', calibrate_magnetometer],
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
        ['Track magnetic sensor', track_magnetic]
    ]
    terminal_menu = TerminalMenu([option[0] for option in options])
    while True:
        choice = terminal_menu.show()
        if type(choice) != int:
            break
        options[choice][1]()  # run the function stored as second element in the chosen row of options


if __name__ == "__main__":
    main()

