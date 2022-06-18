
import numpy as np
import sys
from numpy.linalg import eig, inv
   
def genStdDat(flag):
   a=np.sqrt(2.0)/2.0
   b=np.sqrt(3.0)/3.0
   
   aarr=np.array([

   [ 0, 1, 0],
   [ a, a, 0],
   [ 1, 0, 0],
   [ a,-a, 0],
   [ 0,-1, 0],
   [-a,-a, 0],
   [-1, 0, 0],
   [-a, a, 0],
   [ 0, a, a],
   [ 0, 0, 1],
   [ 0,-a, a],
   [ 0,-a,-a],
   [ 0, 0,-1],
   [ 0, a,-a],
   [ a, 0, a],
   [ a, 0,-a],
   [-a, 0,-a],
   [-a, 0, a]

   ])	
   
   barr=np.array([
   [ 0, 1, 0],
   [ a, a, 0],
   [ 1, 0, 0],
   [ a,-a, 0],
   [ 0,-1, 0],
   [-a,-a, 0],
   [-1, 0, 0],
   [-a, a, 0],
   [ 0, a, a],
   [ 0, 0, 1],
   [ 0,-a, a],
   [ 0,-a,-a],
   [ 0, 0,-1],
   [ 0, a,-a],
   [ a, 0, a],
   [ a, 0,-a],
   [-a, 0,-a],
   [-a, 0, a],

   [ b, b, b],
   [ b, b,-b],
   [ b,-b, b],
   [ b,-b,-b],
   [-b, b, b],
   [-b, b,-b],
   [-b,-b, b],
   [-b,-b,-b]
   ])	  
 
   if flag==0: return aarr
   return barr
   
def defaultParams(): 

   px = 2.0
   py = 2.5
   pz = 2.2
   
   cx = 3.0
   cy = 4.0
   cz = 5.0
   
   rr=10.0
   pp=15.0
   hh=20.0
   
   assymXY=0.00
   assymXZ=0.00
   assymYZ=0.00
   
   pctNoise=0.01
   gain = 1.0
   
   p=np.array([px,py,pz])
   center=np.array([cx,cy,cz])
   rph=np.array([rr,pp,hh])
   assym=np.array([assymXY,assymXZ,assymYZ])
   
   
   
   return (p,center,rph,assym,pctNoise,gain)
   
  
def printMat2( arr, sfmt="%9.4f", title="mat" ):
   ii = len(arr)
   jj = len(arr[0])
   print ( '\n',title,'(',ii,',',jj,')')
#   print "size: ", ii, jj
   for ix in range(ii):
      ss = " "
      for jx in range (jj):
         arij=arr[ix][jx]
         sad =  sfmt % arij
         ss = ss + sad
      print (ss )
   return
  
   
   
   
#right hand rule rotation  
def rotRx(inmat,angdeg):


   # d2r = np.pi / 180.0

   txmat=np.eye(3)
   angrad = np.radians(angdeg)

   cc=np.cos(angrad)
   ss=np.sin(angrad)
   
   txmat[1,1]=cc
   txmat[2,2]=cc
   txmat[1,2]=-ss
   txmat[2,1]= ss

   outmat=np.dot(txmat,inmat)
   return outmat   
   
#  Right hand rule rotation
#  this roll is right wing down
def rotRy(inmat,angdeg):


   # d2r = np.pi / 180.0

   txmat=np.eye(3)
   angrad = np.radians(angdeg)

   cc=np.cos(angrad)
   ss=np.sin(angrad)


   
   txmat[0,0]=cc
   txmat[2,2]=cc
   txmat[0,2]= ss
   txmat[2,0]=-ss

   outmat=np.dot(txmat,inmat)
   return outmat   

def rotRz(inmat,angdeg):

   # d2r = np.pi / 180.0

   txmat=np.eye(3)
   angrad = np.radians(angdeg)

   cc=np.cos(angrad)
   ss=np.sin(angrad)

   
   txmat[0,0]=cc
   txmat[1,1]=cc
   txmat[0,1]=-ss
   txmat[1,0]= ss

   outmat=np.dot(txmat,inmat)
   return outmat
   
def str2float(ss,val): 
   try:
      rc=float(ss)
   except:
      print ('float error:' + ss)
      return val
   return rc
   
def getCmdLine(argv,params,verbose=0): 
   (p,center,rph,assym,pctNoise,gain)=params
   
   token = 'V'
   argLen  = len(argv)
   if argLen < 2:
      print ('-------------------------------------------')
      print ('program [options]')
      print ('\t-p\tPolarizations px,py,pz:  "-p 1.1 1.0 0.9"  ')
      print ('\t-c\tCenter cx,cy,cz:      "-c 1 2 3"')
      print ('\t-r\troll,pitch,heading (deg): "-r 10 20 30"')
      print ('\t-a\tassymmetry XY,XZ,YZ: "-a .001 .002 .003"')
      print ('\t-t\ttoken (in column zero) "-t DMAGACC"')
      print ('\t\t  if token is "-t Y" then print as numpy array')
      print ('\t-n\tNoise : "-n 0.01" is one percent noise')
      print ('\t-g\tGain: "-g 400"')
      print ('-------------------------------------------')
   
   
   kk=1
   while kk < argLen:
      item = argv[kk]
      kk+=1
      if verbose: print( kk,item)
      if len(item) < 2:
         continue
      
      if item[0]!='-':
         continue
      
      item = item.lower()
      
      if item[1] == 'p':
         if argLen - kk < 3: continue
         p[0] = str2float(argv[kk  ],p[0])
         p[1] = str2float(argv[kk+1],p[1])
         p[2] = str2float(argv[kk+2],p[2])
         kk+=3
         
      if item[1] == 'c':
         if argLen - kk < 3: continue
         center[0] = str2float(argv[kk  ],center[0])
         center[1] = str2float(argv[kk+1],center[1])
         center[2] = str2float(argv[kk+2],center[2])

         kk+=3
      
      if item[1] == 'r':
         if argLen - kk < 3: continue
         rph[0] = str2float(argv[kk  ],rph[0])
         rph[1] = str2float(argv[kk+1],rph[1])
         rph[2] = str2float(argv[kk+2],rph[2])

         kk+=3
         
      if item[1] == 'a':
         if argLen - kk < 3: continue
         assym[0] = str2float(argv[kk  ],assym[0])
         assym[1] = str2float(argv[kk+1],assym[1])
         assym[2] = str2float(argv[kk+2],assym[2])

         kk+=3   
      
      if item[1] == 'n':
         if argLen - kk < 1: continue
         pctNoise = str2float(argv[kk],pctNoise)
         kk+=1
         
      if item[1] == 'g':
         if argLen - kk < 1: continue
         gain = str2float(argv[kk],gain)
         kk+=1  

      if item[1] == 't':
         if argLen - kk < 1: continue
         token = argv[kk]
         kk+=1  
   params = (p,center,rph,assym,pctNoise,gain)  
   return (params,token)
   
   
def rotTransMat(params):
   (p,center,rph,assym,pctNoise,gain)=params
   rotMat=np.eye(3)
   r1=rotRy(rotMat,rph[0])
   r2=rotRx(r1,rph[1])
   R=rotRz(r2,rph[2])
   RT=R.T
   # print r1
   # print r2
   # print R
   d=np.diag(p)
   # print d
   DR=np.dot(d,R)
   # print DR
   RTDR=np.dot(RT,DR)
   RTDR[0,1]+= assym[0]
   RTDR[1,0]-= assym[0]
   RTDR[0,2]+= assym[1]
   RTDR[2,0]-= assym[1]
   RTDR[1,2]+= assym[2]
   RTDR[2,1]-= assym[2]
   # print RTDR
   return (R,RTDR)
   
   
def genMagAccSwing():

   # magList=[]
   # accList=[]
   
   magArr=np.zeros([32,3])
   accArr=np.zeros([32,3])
   
   # print magArr
   
   dip = 60.0
   winnowDistance=0.001
   swingPoints = 36
   swingDelta = 360/swingPoints
   
   fwd=np.array([0.0,1.0,0.0])
   
   
   
   acc0 = rotRx(fwd,-90)
   mag0 = rotRx(fwd,-dip)
   
   swarr=np.zeros([swingPoints,3])  
   
   for ix in range(swingPoints):
      ang=ix*swingDelta
      spt=rotRz(mag0,ang)
      # ss=printmagacc(spt,acc,0)
      # swingList.append(ss)
      swarr[ix]=spt
   # printMat2(swarr)   
   
   deltaDeg=45.0
   kk=0
   for ix in range (8):
      rotDeg=ix*deltaDeg
      magr = rotRx(mag0,rotDeg)
      accr = rotRx(acc0,rotDeg)
      # magList.append(magr)
      # accList.append(accr)
      magArr[kk]=magr
      accArr[kk]=accr
      kk+=1
      # print magr,accr
      # mas=printmagacc(magr,accr,0)
      # masg=printmagacc(magr*gain,accr*gain,0)
      # pglist.append(masg)
      # plist.append(mas)
      
   # print magList
   # print accList
   magxz = rotRz(mag0,90)  
   accxz = rotRz(acc0,90)
   for ix in range (8):
      rotDeg=ix*deltaDeg
      magr = rotRy(magxz,rotDeg)
      accr = rotRy(accxz,rotDeg)
      magArr[kk]=magr
      accArr[kk]=accr
      kk+=1
      
      # lm=len(magList)
      # for kk in range(lm):
         # diff = magList[kk] - magr
         # diff2=np.dot(diff,diff)
         # if diff2 < winnowDistance:
            # print ix,kk,diff2
      # print magr,accr
      # printmagacc(magr,accr)
      # magList.append(magr)
      # accList.append(accr)
      
   
   magxy = rotRy(mag0,90)
   accxy = rotRy(acc0,90)
   for ix in range (8):
      rotDeg=ix*deltaDeg
      magr = rotRz(magxy,rotDeg)
      accr = rotRz(accxy,rotDeg)
      magArr[kk]=magr
      accArr[kk]=accr
      kk+=1 
   

        
   deltaDeg=90.0
   
   magt1 = rotRx(mag0,45)
   acct1 = rotRx(acc0,45)
   magt1 = rotRz(magt1,45)
   acct1 = rotRz(acct1,45)
   for ix in range (4):
      rotDeg=ix*deltaDeg
      magr = rotRz(magt1,rotDeg)
      accr = rotRz(acct1,rotDeg)
      magArr[kk]=magr
      accArr[kk]=accr
      kk+=1 
   
   magt2 = rotRx(mag0,135)
   acct2 = rotRx(acc0,135)
   magt2 = rotRz(magt2,45)
   acct2 = rotRz(acct2,45) 
   for ix in range (4):
      rotDeg=ix*deltaDeg
      magr = rotRz(magt2,rotDeg)
      accr = rotRz(acct2,rotDeg)
      magArr[kk]=magr
      accArr[kk]=accr
      kk+=1 
   
   return (swarr,magArr,accArr)   
      
def printmagacc(mag,acc,flag):   
   ss = '%11.5f,%11.5f,%11.5f,%10.5f,%10.5f,%10.5f' % (mag[0],mag[1],mag[2],acc[0],acc[1],acc[2])
   if flag: print (ss)
   return ss
 
# def plotit(mag):
   # xx=mag[:,0]
   # yy=mag[:,1]
   # zz=mag[:,2]
   # fig = plt.figure()
   # ax = fig.add_subplot(111, projection='3d')
   # ax.scatter(xx,yy,zz)
   # delta=800
   # # plt.axis([plotminx,plotmaxx,plotminy,plotmaxy]) 
   # plt.axis([-delta,delta,-delta,delta]) 
   # plt.show()
   
if __name__ == "__main__":
   plist=[]
   pglist=[]
   swingList=[]
   params=defaultParams()
   (p,center,rph,assym,pctNoise,gain)=params
 
   
   (params,token)=getCmdLine(sys.argv,params)
   (p,center,rph,assym,pctNoise,gain)=params
   # print params2
   scaledCenter=center*gain
   print ('p',p)
  
   print ('center',center)
   print ('rph',rph)
   print ('assymmetry',assym)
   print ('noise',pctNoise)
   print ('gain',gain)
   print ('token',token)
   # params=params2
   # noisex = noise*(np.random.standard_normal())
   # noisey = noise*(np.random.standard_normal())
   # x=x + fx + noisex
   # y=y + fy + noisey
   pg=np.sqrt(np.dot(p,p)/3)
   print ('parameter gain',pg)
   
   (R,M)=rotTransMat(params)
   printMat2(R,'%10.4f','Rotation Matrix')
   printMat2(M/M[0,0],'%10.4f','Transform norm1')
   printMat2(M,'%10.4f','Transformation Matrix')
   invM=inv(M)
   printMat2(invM,'%10.4f','Inverted Transform Matrix')
   invM/=invM[0,0]
   printMat2(invM,'%10.4f','invM norm1')
   linearMat=np.reshape(invM,(1,9))[0]
   #printVec( linearMat, "%9.4F", title="linear Matrix values" )
   ss='T,matrix'
   for ix in range(9):
      ss = ss + ',%7.4F' % (linearMat[ix])
   print (ss )
   ss='T,center'
   for ix in range(3):
      ss = ss + ',%9.4F' % (scaledCenter[ix])
   print( ss )
   (swing,magarr,accarr)=genMagAccSwing()
   lm=len(magarr) 
   tempMag=np.dot(M,magarr.T).T
   xt=tempMag[:,0]
   yt=tempMag[:,1]
   zt=tempMag[:,2]
   # rarr = np.sqrt(np.dot(magarr,magarr.T))
   rarr = np.sqrt(xt*xt + yt*yt + zt*zt)
   ravg=np.mean(rarr)
   rstd=np.std(rarr)
   print ('avg/std mag',ravg,rstd)
   magarr/=ravg
   noise=np.zeros([lm,3])
   noise[:,0]=np.random.normal(0.0,pctNoise,lm)
   noise[:,1]=np.random.normal(0.0,pctNoise,lm)
   noise[:,2]=np.random.normal(0.0,pctNoise,lm)
   # print noise
   # exit()
   magarr=magarr+noise
   # printMat2(magarr)
   magarr=np.dot(M,magarr.T).T
   tswing=np.dot(M,swing.T).T
   # printMat2(swing)

   magarr=magarr+center
   tswing=tswing+center
   # printMat2(magarr)
   magarr*=gain
   tswing*=gain
   # plotit(magarr)
   # printMat2(magarr,'%10.1f')
   for ix in range(len(magarr)):
      ss=printmagacc(magarr[ix],accarr[ix],0)
      sp=token + ',' + ss
      print ( sp )
      
   if token=='Y':
      print ('\txyz = np.array([ ')
      for ix in range(len(magarr)):
         ss=printmagacc(magarr[ix],accarr[ix],0)
         sp= '[ ' + ss + ' ],'
         print (sp)
      print ('\t]) ')


   ls = len(swing)
   deltaDeg = 360/ls
   
   for ix in range (ls):
      degTrue = ix*deltaDeg
            
      radObs=np.arctan2(tswing[ix,0],tswing[ix,1])
      degObs=-np.degrees(radObs)
            
      diff = degObs - degTrue
      
      if diff < -180.0: diff += 360.0
      if diff >  180.0: diff -= 360.0
      
      degObs = degTrue + diff
      
      ss="%11.5f,%11.5f,%11.5f,%6.1f,%6.1f,%7.2f" % (
      tswing[ix,0],tswing[ix,1],tswing[ix,2],degTrue,degObs,diff)
      print ('S,' + ss)
   
   exit()   
   
  
   
      
