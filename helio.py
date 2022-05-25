#!/usr/bin/python3

# Heliostat driver - this script should be run as a service

# In the end, this script should provide the remote IoT interface

# During development, it's invoked on command line

import adafruit_fxos8700
import board
from consolemenu import *
from consolemenu.items import *
import csv
from gpiozero import Motor
import logging
import numpy
import random
from scipy.signal import savgol_filter
import time
import vg

# Setup logging
logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d %(funcName)s] %(message)s', datefmt='%Y-%m-%d:%H:%M:%S', level=logging.DEBUG)

# Setup motors
MOTOR_TRAVEL_TIME = 30  # seconds
MOTORH = Motor(26, 20)
MOTORA = Motor(19, 16)

# Setup sensor
i2c = board.I2C()  # uses board.SCL and board.SDA
sensor = adafruit_fxos8700.FXOS8700(i2c)

# Setup calibration data global variable
cal = { 'acxs': [], 'acys': [], 'aczs': [], 'tcxss': [], 'tcyss': [], 'tczss': [] }
NUMBER_OF_SCANS = 1


class Magnetometer(object):
    ''' Magnetometer class with calibration capabilities.

        Parameters
        ----------
        sensor : str
            Sensor to use.
        bus : int
            Bus where the sensor is attached.
        F : float (optional)
            Expected earth magnetic field intensity, default=1.
    '''

    def __init__(self, F=1.):

        self.sensor = FXOS8700()

        # initialize values
        self.F   = F
        self.b   = np.zeros([3, 1])
        self.A_1 = np.eye(3)
        self._calibration_complete = None

    def read(self):
        ''' Get a sample.

            Returns
            -------
            s : list
                The sample in uT, [x,y,z] (corrected if performed calibration).
        '''
        s = np.array(self.sensor.read()).reshape(3, 1)
        s = np.dot(self.A_1, s - self.b)
        return [s[0,0], s[1,0], s[2,0]]

    def launch_calibration(self):
        ''' Performs calibration, actually only the loop collecting the data,
            running in a separate thread. It's monitoring the calibration_complete
            property, set by the finish_calibration method, to stop. 
            The caller must take care of triggering the motion of the sensor and
            of calling finish_calibraiton when done.
        '''

        if self._calibration_complete == True:
            logging.error("Trying to launch calibration before previous one finished!")
            return
        self._calibration_s = []
        self._calibration_n = 0
        self._calibration_complete = False
        Thread(target=self.__collect_calibration_data).start()


    def __collect_calibration_data(self):
        while self.calibration_complete == False:
            self._calibration_s.append(self.sensor.read())
            self._calibration_n += 1


    def finish_calibration(self):
        ''' Finishes the calibration, by setting the calibration_complete property
            to finish data collection by the launch_calibration thread and then
            calculating the calibration result
        '''

        # finish reading sensor data
        self._calibration_complete = True
        s = self._calibration_s
        n = self._calibration_n

        # ellipsoid fit
        s = np.array(s).T
        M, n, d = self.__ellipsoid_fit(s)

        # calibration parameters
        # note: some implementations of sqrtm return complex type, taking real
        M_1 = linalg.inv(M)
        self.b = -np.dot(M_1, n)
        self.A_1 = np.real(self.F / np.sqrt(np.dot(n.T, np.dot(M_1, n)) - d) *
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
        self.A_1 = np.load('magcal_A1.npy')
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


class FXOS8700(object):
    ''' FXOS8700 Simple Driver.
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

        number_of_readouts = 48

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


def calibrate_magnetometer():
    global ms
    ms.launch_calibration()  # starts reading
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
    logging.info("Stop magnetometer calibration")
    
    
def save_magnetometer_calibration()
    global ms



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
        cal['tcxss'].append(savgol_filter(tcxs, 9, 1))
        cal['tcyss'].append(savgol_filter(tcys, 9, 1))
        cal['tczss'].append(savgol_filter(tczs, 9, 1))
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
            aangle = vg.angle(numpy.array([racx, racy, racz]), numpy.array([acx, acy, acz]))
            tangle = vg.angle(numpy.array([rtcx, rtcy, rtcz]), numpy.array([tcx, tcy, tcz]))
            print("azimut: {:4.1f} - heading: {:4.1f}".format(aangle, tangle), end="\r")
            time.sleep(0.1)
    except KeyboardInterrupt:
        return



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
    global ms
    ms = Magnetometer(50)  # 50 is just a general scaling factor for the readouts
    menu = ConsoleMenu("Heliostat", "Control Center")
    menu.append_item(FunctionItem("Calibrate magnetometer", calibrate_magnetometer))
    menu.append_item(FunctionItem("Save magnetometer calibration to file", save_magnetometer_calibration))
    menu.append_item(FunctionItem("Load calibration from file", load_magnetometer_calibration))
    menu.append_item(FunctionItem("Calibrate", calibrate))
    menu.append_item(FunctionItem("Save calibration to file", save_calibration))
    menu.append_item(FunctionItem("Load calibration from file", load_calibration))
    menu.append_item(FunctionItem("Track angular distance to random calibration point", track_random))
    menu.show()


if __name__ == "__main__":
    main()

