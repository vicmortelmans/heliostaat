

# Sample calculations of magnetic corrections using iterative techniques
# Both precision and symmetric solutions are shown
# No linear algebra libraries are required. Simple matrix inversion routine is included
# No eigenvalue-eigenvector routines are required
# Works with Python Version 2.7
#
# Copyright 2020, Tom Judd 

import numpy as np
import sys
import os
import matplotlib.pyplot as plt

iterations=7 #global Number of iterations 
verbose = 0  #global Prints more when verbose=1
plotMe  = 1  #global Plot of original vs corrected compass errors

# ============================= 
# simple matrix inversion
# assume well behaved data so no pivoting is used here

def mat_invert(A,epsilon=1E-10):
   
   N = len(A)

   Ainv=np.eye(N)
   
   for ii in range(N):
      alpha = A[ii][ii]
      if abs(alpha) < epsilon: return (Ainv,-1)
      
      for jj in range(N):
         A[ii][jj] =  A[ii][jj]/alpha
         Ainv[ii][jj] =  Ainv[ii][jj]/alpha 
      
      for kk in range(N):
         if kk==ii: continue
         beta = A[kk][ii]
         for jj in range(N):
            A[kk][jj]=A[kk][jj] - beta * A[ii][jj]
            Ainv[kk][jj]=Ainv[kk][jj] - beta * Ainv[ii][jj]
  
   return (Ainv,1)

 
# =============================  

def printVec( vec, sfmt="%9.4F", title="vec" ):
   ii = len(vec)
#   print "size: ", ii
   print( '\n',title,'(',ii,')')
   ss = " "
   for ix in range(ii):
      ari=np.real(vec[ix])
#      print ari
      sad =  sfmt % ari
      ss = ss + sad
   print ( ss )  
   return
 

# =============================  
  
def applyParams12(xyz,params):
   # first three of twelve parameters are the x,y,z offsets
   ofs=params[0:3]
   # next nine are the components of the 3x3 transformation matrix
   mat=np.reshape(params[3:12],(3,3))
   # subtract ofs
   xyzcentered=xyz-ofs
 
   xyzout=np.dot(mat,xyzcentered.T).T
   
   return xyzout
   
      

# =============================  
  
   
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

# =============================  
   

# numeric version of calculating partial derivatives
   
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

   
# =============================  


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

# =============================  
 
def ofsMatToParam9(ofs,mat,params):

   params[0:3]=ofs
   
   params[3]=mat[0,0]
   params[4]=mat[1,1]
   params[5]=mat[2,2]

   params[6]=mat[0,1]
   params[7]=mat[0,2]
   params[8]=mat[1,2]
  
   return params


# =============================  
   
#mode: 1=precision, use accels

def errFn(mag,acc,target,params,mode):
   if mode == 1: return magDotAccErr(mag,acc,target,params)
   return radiusErr(mag,target,params)

# =============================  
   
def radiusErr(mag,target,params):
   #offset and transformation matrix from parameters
   (ofs,mat)=param9toOfsMat(params)
   
   #subtract offset, then apply transformation matrix
   mc=mag-ofs
   mm=np.dot(mat,mc)

   radius = np.sqrt(mm[0]*mm[0] +mm[1]*mm[1] + mm[2]*mm[2] )
   err=target-radius
   return err
# =============================  
   
def magDotAccErr(mag,acc,mdg,params):
   #offset and transformation matrix from parameters
   ofs=params[0:3]
   mat=np.reshape(params[3:12],(3,3))
   #subtract offset, then apply transformation matrix
   mc=mag-ofs
   mm=np.dot(mat,mc)
   #calculate dot product from corrected mags
   mdg1=np.dot(mm,acc)
   err=mdg-mdg1
   return err
# =============================  

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
# =============================  
   
def normalize3(xyz):
   x=xyz[:,0]
   y=xyz[:,1]
   z=xyz[:,2]
   rarr = np.sqrt(x*x + y*y + z*z)
   ravg=np.mean(rarr)
   xyzn=xyz/ravg
   return (xyzn,ravg)
# =============================  
    
   
def errorEstimate(magN,accN,target,params):
   err2sum=0
   nsamp=len(magN)
   for ix in range(nsamp):
      err=magDotAccErr(magN[ix],accN[ix],target,params)
      err2sum += err*err
      # print "%10.6f" % (err)
   sigma=np.sqrt(err2sum/nsamp)  
   return sigma
# =============================  
   
def errorEstimateSymmetric(mag,target,params):
   err2sum=0
   nsamp=len(mag)
   for ix in range(nsamp):
      err=radiusErr(mag[ix],target,params)
      err2sum += err*err
      # print "%10.6f" % (err)
   sigma=np.sqrt(err2sum/nsamp)  
   return sigma  
# =============================  
   
def printParams(params,fmt='%10.6f',scale=1.0):

   ofs=np.array(params[0:3])
      # next nine are the components of the 3x3 transformation matrix
   mat=np.array(np.reshape(params[3:12],(3,3)))
   ofs*=scale
   mat*=scale
   # s0 = '"' + fmt + '"  "' + fmt + '"  "' +  fmt +  '"'
   s0 =  fmt + '  ' + fmt + '  ' +  fmt +  ' '
   # print s0
   ss=s0 % (ofs[0],ofs[1],ofs[2])
   print(ss,'\n')
   for ix in range(3):
      ss=s0 % (mat[ix,0],mat[ix,1],mat[ix,2])
      print(ss)
   print ('\n'   )
# =============================  
  
     
def printMat2( arr, sfmt="%9.4f", title="mat" ):
   ii = len(arr)
   jj = len(arr[0])
   print ('\n',title,'(',ii,',',jj,')')
#   print "size: ", ii, jj
   for ix in range(ii):
      ss = " "
      for jx in range (jj):
         arij=arr[ix][jx]
         # print sfmt
         sad =  sfmt % (arij)
         ss = ss + sad
      print (ss)
   return
# =============================  
   
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
# =============================
     
def ellipsoid_iterate_symmetric(mag,verbose):
      
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
   nLoops=iterations

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
   
    
   
# =============================
     
def ellipsoid_iterate(mag,accel,verbose):
      
   # magCorrected=copy.deepcopy(mag)
   # Obtain an estimate of the center and radius
   # For well distributed samples, the average of all points is sufficient
   
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
   if verbose: print ('Initial Sigma',sigma)
   
   # pre allocate the data.  We do not actually need the entire
   # D matrix ( a nSamples x 12 matrix ) if we calculate DTD (a 12x12 matrix) within the sample loop
   # Also DTE (dimension 12) can be calculated on the fly. 
  
   D=np.zeros([nSamples,12])
   E=np.zeros(nSamples)
   
   #If numeric derivatives are used, this step size works with normalized data.
   step=np.ones(12)  
   step/=5000
   
   #Fixed number of iterations for testing.  In production you check for convergence
   
   nLoops=iterations

   for iloop in range(nLoops):
      # Numeric or analytic partials each give the same answer
      for ix in range(nSamples):
         # (f0,pdiff)=numericPartialRow(magScaled[ix],accNorm[ix],avgDot,params,step,1)
         (f0,pdiff)=analyticPartialRow(magScaled[ix],accNorm[ix],avgDot,params)
         E[ix]=f0
         D[ix]=pdiff
      #Use the pseudo-inverse   
      DT=D.T
      DTD=np.dot(DT,D)
      DTE=np.dot(DT,E)
      invDTD=np.linalg.inv(DTD)
      deltas=np.dot(invDTD,DTE)

      #negative sign because of the way we defined the derivatives
      p2=params + deltas
      
      
      sigma = errorEstimate(magScaled,accNorm,avgDot,p2)
      
      # add some checking here on the behavior of sigma from loop to loop
      # if satisfied, use the new set of parameters for the next iteration

      params=p2
      
      # recalculste gain (magR) and target dot product
      # not strictly required, the symmetric algorithm does not renormalice each loop
      
      magCorrected=applyParams12(magScaled,params)
      (mc,mcR)=normalize3(magCorrected)
      (avgDot,stdDot)=mgDot(mc,accNorm)
      magR *= mcR
      magScaled=mag/magR
      
      if verbose: 
         print ('iloop',iloop,'sigma',sigma)
   
   return (params,magR)
 
# =============================

def getCmdLine(argv,verbose): 
   fn=''
# file data is of form token,magx,magy,magz,accx,accy,accz,. . .    
   mcol=1
   acol=4
   token='V'
   argLen  = len(argv)
   if argLen < 2:
      print ('program [options]')
      print ('\t-f\tinput file name')
      
   kk=1
   while (argLen - kk) > 1:  #data comes in pairs '-z itemz'
      item = argv[kk]
      if verbose: print( kk,item)
      if len(item) < 2:
         kk+=1
         continue
      
      if item[0]!='-':
         kk+=1
         continue
      
      item = item.lower()
      
      if item[1] == 'f':
         fn=argv[kk+1]
               
      if verbose: print( kk+1,argv[kk+1]   )
      kk+=2
 
      
   return (fn,mcol,acol,token) 
   
# =============================


def str2float(ss,val): 
   try:
      rc=float(ss)
   except:
      return val #silent error
   return rc
   
# =============================
# Read input file generated by gendat.py

def readFileData(cmdlineStuff):

                  #items read from the file:
   dataList=[]    #observed samples
   swingList=[]   #level data points: 'S,mx,my,mz,...' simulating calibration data taked around a circle
   tcenter=[]     #true center, for comparison with calculations
   tmatrix=[]     #true transformation matrix, for comparison with calculations
   
   
   rc=(False,0,0,0,0,0) # return (ok,mag,accel,swingList,true center, true xform matrix )
   
   (fn,magix,accix,tok)=cmdlineStuff
   
   if not os.path.isfile(fn) :
      print('Could not open input file ' + fn)
      return rc
   try:
      f = open(fn,'rt')
      lines = f.readlines()
      f.close()
   except:
      print ('Could not open input file ' + fn)
      return rc
   
   for aLine in lines:
      aLine=aLine.strip()
      toks=aLine.split(',')
      
      
      if toks[0] == tok: # all samples have an identifying token
         if len(toks) < 7: continue # token,mx,my,mz,ax,ay,az = at least seven items 
         for ix, anItem in enumerate(toks):
            toks[ix] = str2float(anItem,0)
         dataList.append(toks[1:]) # start at 1, not 0, to eliminate token
         
      if toks[0] == 'S': # calibration points taken while the device is swung around 360 deg 
         if len(toks) < 4: continue # S,mx,my,mz = at least 4 samples
         for ix, anItem in enumerate(toks):
            toks[ix] = str2float(anItem,0)
         swingList.append(toks[1:4]) # drop the 'S' and save only mx,my,mz
         
      if toks[0] == 'T': # truth data: center, and transformation matrix 
      
         if len(toks) < 5: continue # 'T,center,cx,cy,cz' = at least 5 tokens
         if toks[1]=='center':
            for ix, anItem in enumerate(toks):
               tflt = str2float(anItem,0)
               if ix>1: tcenter.append(tflt) # start at index 2 = cx
            
            
         if len(toks) < 11: continue # 'T,matrix,m11,m12,m13,m21,m22,m23,m31,m32,m33'
         if toks[1]=='matrix':
            for ix, anItem in enumerate(toks):
               tflt = str2float(anItem,0)
               if ix>1: tmatrix.append(tflt) #saved here as 9 element linear array, not 3x3 
         
   
   if len(dataList) < 16: return rc  
   if len(swingList) < 8: return rc
   if len(tcenter) < 3: return rc
   if len(tmatrix) < 9: return rc
   
   dataArr=np.array(dataList)
   swingArr=np.array(swingList)

   tcenter=np.array(tcenter)
   tmatrix=np.reshape(np.array(tmatrix),(3,3)) #now a 3x3
   
   mag = dataArr[:,:3] #all rows, columns 0..2
   acc = dataArr[:,3:6] #all rows, columns 3..5
   
   return (True,mag,acc,swingArr,tcenter,tmatrix)
   

# =============================
def param9toParam12(p9):
   (ofs,mat)=param9toOfsMat(p9)
   p12=np.array(np.zeros(12))
   p12[:3]=ofs
   p12[3:12]=np.reshape(mat,(1,9))
   return p12
   
# =============================
def calculateAngleError(mag,params,magScale,swingPoints):
   swingCorrected=applyParams12(swingPoints/magScale,params)
   slen=len(swingPoints)   
   deltaDeg = 360/slen
   errO=np.array(np.zeros(slen))
   errC=np.array(np.zeros(slen))
   degArr=np.array(np.zeros(slen))
   
   for ix in range (slen):
      degTrue = ix*deltaDeg
      degArr[ix]=degTrue
      
      radObs=np.arctan2(swingPoints[ix,0],swingPoints[ix,1])
      degObs=-np.degrees(radObs)
      
      radCorrected=np.arctan2(swingCorrected[ix,0],swingCorrected[ix,1])
      degCorrected=-np.degrees(radCorrected)      
      
      diffO = degObs - degTrue
      
      if diffO < -180.0: diffO += 360.0
      if diffO >  180.0: diffO -= 360.0
      
      degObs = degTrue + diffO
      
      diffC = degCorrected - degTrue
      
      if diffC < -180.0: diffC += 360.0
      if diffC >  180.0: diffC -= 360.0
      
      degCorrected = degTrue + diffC
      
      errO[ix]=diffO
      errC[ix]=diffC

   avgO=np.mean(errO)   
   avgC=np.mean(errC)   
   errO-=avgO
   errC-=avgC
   stdO=np.std(errO)
   stdC=np.std(errC)
  
   return (stdO,stdC,errO,errC)

   
# =============================
def plotit360(dataArr,aTitle):
   
   (nvec,vlen)=dataArr.shape
  
   delta = int(360/vlen)
   
   xx=np.array(range(0,360,delta))
     
   fig, ax = plt.subplots()
   
   #expecting at least 3 arrays for ploting.
   ax.plot(xx,dataArr[0],'r')
   ax.plot(xx,dataArr[1],'b')
   ax.plot(xx,dataArr[2],'g')
   
   ax.set(xlabel='True Angle (deg)', ylabel='Compass Error (deg)', title=aTitle)
   ax.grid()
   plt.axis([0,360,-15,15]) 
   plt.show()

# =============================


if __name__ == "__main__":

   xyz= np.array( [
[6.7585,  12.4687,  10.6322],
[8.1433,  11.8438,  10.5958],
[8.6387,  10.1855,  10.6127],
[7.9088,   8.4037,  10.6471],
[6.4050,   7.6305,  10.6705],
[4.9783,   8.1582,  10.7178],
[4.4923,   9.8695,  10.6851],
[5.2169,  11.6423,  10.6559],
[6.6942,  11.7483,  12.1768],
[6.4671,   9.9641,  12.8641],
[6.4039,   8.2774,  12.2121],
[6.5194,   8.3513,   9.1169],
[6.6307,  10.0678,   8.4697],
[6.6970,  11.7103,   9.0651],
[7.9516,  10.1011,  12.1489],
[8.0684,  10.1892,   9.0646],
[5.1151,   9.9348,   9.1366],
[5.0583,   9.9015,  12.2513]

]   )

   xyz400 = np.array([
[   321.13761,  592.68208, -110.92169,   0.00000,   0.00000,  -1.00000 ],
[   369.57750,  820.62965,  -33.77537,   0.00000,   0.70711,  -0.70711 ],
[   353.30059,  710.18674,   99.50110,   0.00000,   1.00000,   0.00000 ],
[   309.06748,  296.01928,  231.44519,   0.00000,   0.70711,   0.70711 ],
[   239.27049, -177.83822,  273.04763,   0.00000,   0.00000,   1.00000 ],
[   192.47963, -434.44221,  198.11776,   0.00000,  -0.70711,   0.70711 ],
[   195.28058, -304.18968,   53.81068,   0.00000,  -1.00000,   0.00000 ],
[   259.38827,  110.18309,  -76.44162,   0.00000,  -0.70711,  -0.70711 ],
[   100.92073,  221.00200,  -72.69141,  -0.00000,   0.00000,  -1.00000 ],
[   -54.46045,  149.99361,   30.56772,  -0.70711,   0.00000,  -0.70711 ],
[   -21.70442,   92.71619,  170.56060,  -1.00000,   0.00000,   0.00000 ],
[   187.70107,  115.00077,  253.07234,  -0.70711,   0.00000,   0.70711 ],
[   454.83761,  180.06335,  238.11447,  -0.00000,   0.00000,   1.00000 ],
[   619.01148,  266.21554,  129.16750,   0.70711,   0.00000,   0.70711 ],
[   582.25116,  310.30544,  -10.64292,   1.00000,   0.00000,   0.00000 ],
[   370.97056,  295.59823,  -93.93400,   0.70711,   0.00000,  -0.70711 ],
[    18.89031,  429.91097,   42.37785,  -1.00000,   0.00000,  -0.00000 ],
[   -86.10727,  -54.91491,  100.34645,  -0.70711,  -0.70711,  -0.00000 ],
[    28.97991, -389.40055,  145.12057,  -0.00000,  -1.00000,  -0.00000 ],
[   292.69263, -372.17646,  151.21780,   0.70711,  -0.70711,  -0.00000 ],
[   545.15406,  -42.78629,  117.71466,   1.00000,  -0.00000,  -0.00000 ],
[   653.36751,  455.18348,   62.37725,   0.70711,   0.70711,  -0.00000 ],
[   534.53481,  800.83089,   14.29840,   0.00000,   1.00000,  -0.00000 ],
[   277.36174,  776.81465,    9.01628,  -0.70711,   0.70711,  -0.00000 ],
[    99.67700,  606.21526,  -21.33528,  -0.50000,   0.50000,  -0.70711 ],
[   -27.67416, -262.68674,   83.14308,  -0.50000,  -0.50000,  -0.70711 ],
[   460.00349, -147.02142,   82.40870,   0.50000,  -0.50000,  -0.70711 ],
[   585.50705,  715.83148,  -18.04456,   0.50000,   0.50000,  -0.70711 ],
[   244.30505,  248.30528,  236.85131,  -0.50000,   0.50000,   0.70711 ],
[   201.29068,    7.70060,  265.53074,  -0.50000,  -0.50000,   0.70711 ],
[   328.19573,   35.07879,  266.39908,   0.50000,  -0.50000,   0.70711 ],
[   360.17339,  248.83504,  240.78515,   0.50000,   0.50000,   0.70711 ],
        ])

   magAccel400x=np.array([
[   44.47,  247.49, -312.31,   0.0000,   0.0000,  -1.0000],
[   48.33,  446.03, -102.17,   0.0000,   0.7071,  -0.7071],
[   52.98,  391.72,  175.80,   0.0000,   1.0000,   0.0000],
[   45.95,  120.61,  352.99,   0.0000,   0.7071,   0.7071],
[   34.79, -210.67,  326.69,   0.0000,   0.0000,   1.0000],
[   29.55, -402.91,  112.03,   0.0000,  -0.7071,   0.7071],
[   28.07, -350.69, -160.51,   0.0000,  -1.0000,   0.0000],
[   39.46,  -85.62, -341.82,   0.0000,  -0.7071,  -0.7071],
[ -166.04,   22.71, -302.64,  -0.0000,   0.0000,  -1.0000],
[ -352.17,   15.41,  -88.25,  -0.7071,   0.0000,  -0.7071],
[ -308.88,    0.31,  188.51,  -1.0000,   0.0000,   0.0000],
[  -62.78,    6.96,  356.68,  -0.7071,   0.0000,   0.7071],
[  243.77,   15.36,  324.14,  -0.0000,   0.0000,   1.0000],
[  427.44,   27.25,  105.51,   0.7071,   0.0000,   0.7071],
[  389.16,   33.52, -174.49,   1.0000,   0.0000,   0.0000],
[  141.61,   33.56, -340.88,   0.7071,   0.0000,  -0.7071],
[ -302.41,  227.87,    1.93,  -1.0000,   0.0000,  -0.0000],
[ -351.93, -100.40,   10.62,  -0.7071,  -0.7071,  -0.0000],
[ -169.24, -361.41,   19.78,  -0.0000,  -1.0000,  -0.0000],
[  133.51, -399.85,   21.87,   0.7071,  -0.7071,  -0.0000],
[  381.00, -190.13,   16.78,   1.0000,  -0.0000,  -0.0000],
[  429.07,  146.07,    7.42,   0.7071,   0.7071,  -0.0000],
[  250.70,  403.59,   -2.33,   0.0000,   1.0000,  -0.0000],
[  -54.22,  436.38,   -4.19,  -0.7071,   0.7071,  -0.0000],
[ -228.17,  315.40,  -96.63,  -0.5000,   0.5000,  -0.7071],
[ -243.17, -279.36,  -76.87,  -0.5000,  -0.5000,  -0.7071],
[  311.00, -266.72,  -76.13,   0.5000,  -0.5000,  -0.7071],
[  323.65,  332.63,  -94.30,   0.5000,   0.5000,  -0.7071],
[  -29.22,   83.71,  355.76,  -0.5000,   0.5000,   0.7071],
[  -31.41,  -72.17,  357.86,  -0.5000,  -0.5000,   0.7071],
[  111.05,  -66.95,  358.39,   0.5000,  -0.5000,   0.7071],
[  117.51,   86.81,  355.55,   0.5000,   0.5000,   0.7071]
])  
   
 
   #above arrays are used if there is no file specified on the command line
   
   magAccel=xyz400
   
   
   
   cmdlineStuff= getCmdLine(sys.argv,verbose)   
   
   (fn,magix,accix,tok)=cmdlineStuff
   (ok,mag,acc,swingPoints,tcenter,tmatrix)=readFileData(cmdlineStuff)
   
   if ok:
      printVec(tcenter,'%10.2f','Truth Center ')
      printMat2(tmatrix,"%10.4f",'Truth Transform Matrix (normalized)')
      
      print ('\n============================')
      print ('Precision calculation on data in file ' + fn )
      (params,magScale) = ellipsoid_iterate(mag,acc,verbose)
      ofs=params[0:3]*magScale
      printVec(ofs,'%10.2f','Calculated center,Precision')
      
      mat=np.reshape(params[3:12],(3,3))
      printMat2(mat/mat[0,0],"%10.4f",'Calculated Transform Matrix, Precision')
      (errO,errPrecision,errInArr,errOutArrP)=calculateAngleError(mag,params,magScale,swingPoints)
      
      print ('\n============================')

      print ('Symmetric calculation on data in file ' + fn )
      (params9,magScale) = ellipsoid_iterate_symmetric(mag,verbose)
      (ofs,mat)=param9toOfsMat(params9)
      ofs=ofs*magScale
      printVec(ofs,'%10.2f','Calculated center,Symmetric')
      
      printMat2(mat/mat[0,0],"%10.4f",'Calculated Transform Matrix, Symmetric')
      params12=param9toParam12(params9)

      (errO,errSymmetric,errInArr,errOutArrS)=calculateAngleError(mag,params12,magScale,swingPoints)
      pldata=np.array([errInArr,errOutArrS,errOutArrP])

      s0='UnCalibrated Compass:Red, Symmetric Cal:Blue, Precision Cal:Green\n' 
      s1='Corresponding RMS errors -- %7.2f   %7.2f   %7.2f' % (errO,errSymmetric,errPrecision)
      if plotMe: plotit360(pldata,s0+s1)
    

      print ('\n============================')
      print ('Degrees error')
      ss = "\nOriginal : %10.2f"  % (errO)
      ss1 = "\n\nAfter Calibration :\n\nSymmetric: %10.2f" % (errSymmetric)
      ss=ss+ss1
      ss1 = "\nPrecision: %10.2f" % (errPrecision)
      ss=ss+ss1

      print (ss)
      exit()
      
      
   #==========   
   # if not using an input file then use the hard coded arrays.
   
   magAccel=xyz400
   mag=magAccel[:,0:3]
   acc=magAccel[:,3:6]
  
   
   #==========
   print ('\nPrecision calculation on xyz400 data\n')
   (params,magScale) = ellipsoid_iterate(mag,acc,1)
   printParams(params)
   ofs=params[0:3]*magScale
   printVec(ofs,'%10.2f','Offsets ')
   mat=np.reshape(params[3:12],(3,3))
   printMat2(mat/mat[0,0],"%10.4f",'Transform Matrix norm1')
   #=========
   # exit()
   print( '\nSymmetric calculation on xyz400 data\n')

   # print magScale
   (params9,magScale) = ellipsoid_iterate_symmetric(mag,1)
   (ofs,mat)=param9toOfsMat(params9)
   printVec(ofs*magScale,'%10.2f','Offsets ')
   printMat2(mat/mat[0,0],"%10.4f",'Transform Matrix norm1')
   #==========

   magAccel=magAccel400x
   mag=magAccel[:,0:3]
   acc=magAccel[:,3:6]

   #==========
   print ('\nPrecision calculation on magAccel400x data\n')

   (params,magScale) = ellipsoid_iterate(mag,acc,1)
   printParams(params)
   ofs=params[0:3]*magScale
   printVec(ofs,'%10.2f','Offsets ')
   mat=np.reshape(params[3:12],(3,3))
   printMat2(mat/mat[0,0],"%10.4f",'Transform Matrix norm1')
   #=========
 
   print ('\nSymmetric calculation on magAccel400x data\n')

   (params9,magScale) = ellipsoid_iterate_symmetric(mag,1)
   (ofs,mat)=param9toOfsMat(params9)
   printVec(ofs*magScale,'%10.2f','Offsets ')
   printMat2(mat/mat[0,0],"%10.4f",'Transform Matrix norm1')
   #==========

   exit()
   
   
   
   
