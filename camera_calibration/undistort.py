# https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_calib3d/py_calibration/py_calibration.html#calibration
# run as python3 undistort.py <image filename>
import numpy as np
import cv2
import sys

DIM=(2592, 1944)
mtx=np.array([[1513.35202186325, 0.0, 1381.794375023546], [0.0, 1514.809082655238, 1022.1313014429818], [0.0, 0.0, 1.0]])
dist=np.array([[-0.3293226333311312, 0.13030355339675337, 0.00020716954584170977, -0.00032937886446441326, -0.027128518075549755]])

def undistort(img_path, balance=0.0, dim2=None, dim3=None):

    img = cv2.imread(img_path)
    dim1 = img.shape[:2][::-1]  #dim1 is the dimension of input image to un-distort

    assert (dim1[0] == DIM[0]) and (dim1[1] == DIM[1]), "Image to undistort needs to have same dimensions as the ones used in calibration"

    if not dim2:
        dim2 = dim1

    if not dim3:
        dim3 = dim1

    h,  w = img.shape[:2]
    newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1.0,(w,h))

    # undistort
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

    # crop the image (increasing the cropping by 20%)
    x,y,w,h = roi
    b = int(y+h+.2*h)
    o = int(y-.2*h)
    l = int(x-.2*w)
    r = int(x+w+.2*w)
    dst = dst[o:b, l:r]
    cv2.imwrite(img_path + '_undistorted.png',dst)

    cv2.imshow("undistorted", dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':

    for p in sys.argv[1:]:
        undistort(p, balance=0.8)
