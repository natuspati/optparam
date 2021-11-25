# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 10:12:46 2021

@author: bekdulnm
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R
from classes import *
import csv
import time

if __name__ == "__main__":
    
    lst = np.loadtxt("forplotting1.csv", delimiter=',')
    
    j = 0
    itlst = []
    flst = []
    glst = []
    thetalin = []
    philin = []
    rlin = []
    thetalou = []
    philou = []
    rlou = []
    thetarin = []
    phirin = []
    rrin = []
    thetarou = []
    phirou = []
    rrou = []
    focx = []
    focy = []
    focz = []
    mx = []
    my = []
    tx = []
    ty = []
    tz = []
    rx = []
    ry = []
    rz = []
    
    
    for i in np.linspace(0,lst.shape[0]-1,200,dtype=int):
        j += 1
        itlst.append(i)
        flst.append(lst[i,1])
        glst.append(lst[i,0])
        thetalin.append(lst[i,2])
        philin.append(lst[i,3])
        rlin.append(lst[i,4])
        thetalou.append(lst[i,5])
        philou.append(lst[i,6])
        rlou.append(lst[i,7])
        thetarin.append(lst[i,8])
        phirin.append(lst[i,9])
        rrin.append(lst[i,10])
        thetarou.append(lst[i,11])
        phirou.append(lst[i,12])
        rrou.append(lst[i,13])
        focx.append(lst[i,14])
        focy.append(lst[i,15])
        focz.append(lst[i,16])
        mx.append(lst[i,17])
        my.append(lst[i,18])
        tx.append(lst[i,19])
        ty.append(lst[i,20])
        tz.append(lst[i,21])
        rx.append(lst[i,22])
        ry.append(lst[i,23])
        rz.append(lst[i,24])
        
    itlst = np.array([itlst])
    flst = np.array([flst])
    glst = np.array([glst])
    thetalin = np.array([thetalin])
    philin = np.array([philin])
    rlin = np.array([rlin])
    thetalou = np.array([thetalou])
    philou = np.array([philou])
    rlou = np.array([rlou])
    thetarin = np.array([thetarin])
    phirin = np.array([phirin])
    rrin = np.array([rrin])
    thetarou = np.array([thetarou])
    phirou = np.array([phirou])
    rrou = np.array([rrou])
    focx = np.array([focx])
    focy = np.array([focy])
    focz = np.array([focz])
    mx = np.array([mx])
    my = np.array([my])
    tx = np.array([tx])
    ty = np.array([ty])
    tz = np.array([tz])
    rx = np.array([rx])
    ry = np.array([ry])
    rz = np.array([rz])
    
    
    
    
    thetalin = thetalin*180/np.pi
    thetalou = thetalou*180/np.pi
    thetarin = thetarin*180/np.pi
    thetarou = thetarou*180/np.pi
    
    philin = philin*180/np.pi
    philou = philou*180/np.pi
    phirin = phirin*180/np.pi
    phirou = phirou*180/np.pi
    
    
    eax = np.copy(rx)
    eay = np.copy(ry)
    eaz = np.copy(rz)
    for i, elem in enumerate(rx):
        rv = np.array([rx[0,i],ry[0,i],rz[0,i]])
        ri = R.from_rotvec(rv)
        ea = ri.as_euler('xyz', degrees=True)
        eax[i] = ea[0]
        eay[i] = ea[1]
        eaz[i] = ea[2]
    
    plt.close("all")
    plt.figure(1)
    plt.scatter(itlst,flst,s=80,facecolors='none',color='C0')
    plt.title('Value of objective function over iterations')
    plt.xlabel('iteration number')
    plt.ylabel('obj. function value')
    plt.grid(b=True, which='both')
    
    plt.figure(2)
    plt.scatter(itlst,glst,s=80,facecolors='none',color='C0')
    plt.title('Value of norm of the gradient over iterations')
    plt.xlabel('iteration number')
    plt.ylabel('norm of the gradient')
    plt.grid(b=True, which='both')
    
    plt.figure(3)
    plt.scatter(itlst,thetalin,s=80,facecolors='none',color='C0')
    plt.scatter(itlst,thetalou,s=80,facecolors='none',color='C1')
    plt.scatter(itlst,thetarin,s=80,facecolors='none',color='C2')
    plt.scatter(itlst,thetarou,s=80,facecolors='none',color='C3')
    plt.title(r'$\theta$ over iterations')
    plt.xlabel('iteration number')
    plt.ylabel('degrees')
    plt.grid(b=True, which='both')
    plt.legend(['left inner','left outer','right inner', 'right outer'])
    
    plt.figure(4)
    plt.scatter(itlst,philin,s=80,facecolors='none',color='C0')
    plt.scatter(itlst,philou,s=80,facecolors='none',color='C1')
    plt.scatter(itlst,phirin,s=80,facecolors='none',color='C2')
    plt.scatter(itlst,phirou,s=80,facecolors='none',color='C3')
    plt.title(r'$\phi$ over iterations')
    plt.xlabel('iteration number')
    plt.ylabel('degrees')
    plt.grid(b=True, which='both')
    plt.legend(['left inner','left outer','right inner', 'right outer'])
    
    plt.figure(5)
    plt.scatter(itlst,rlin,s=80,facecolors='none',color='C0')
    plt.scatter(itlst,rlou,s=80,facecolors='none',color='C1')
    plt.scatter(itlst,rrin,s=80,facecolors='none',color='C2')
    plt.scatter(itlst,rrou,s=80,facecolors='none',color='C3')
    plt.title(r'$r$, position of mirror plane over iterations')
    plt.xlabel('iteration number')
    plt.ylabel('mm')
    plt.grid(b=True, which='both')
    plt.legend(['left inner','left outer','right inner', 'right outer'])
    
    plt.figure(6)
    plt.scatter(itlst,focx,s=80,facecolors='none',color='C0')
    plt.scatter(itlst,focy,s=80,facecolors='none',color='C1')
    plt.scatter(itlst,focz,s=80,facecolors='none',color='C2')
    plt.title(r'$T_{lens}$, position of lens over iterations')
    plt.xlabel('iteration number')
    plt.ylabel('mm')
    plt.grid(b=True, which='both')
    plt.legend([r'$T_x$',r'$T_y$',r'$T_z$'])
    
    plt.figure(7)
    plt.scatter(itlst,mx,s=80,facecolors='none',color='C0')
    plt.scatter(itlst,my,s=80,facecolors='none',color='C1')
    plt.title(r'$m_x$ and $m_y$ pixel scalings over iterations')
    plt.xlabel('iteration number')
    plt.ylabel('mm/px')
    plt.grid(b=True, which='both')
    plt.legend([r'$m_x$',r'$m_y$'])
    
    plt.figure(8)
    plt.scatter(itlst,tx,s=80,facecolors='none',color='C0')
    plt.scatter(itlst,ty,s=80,facecolors='none',color='C1')
    plt.scatter(itlst,tz,s=80,facecolors='none',color='C2')
    plt.title(r'$T_{target}$, position of target over iterations')
    plt.xlabel('iteration number')
    plt.ylabel('mm')
    plt.grid(b=True, which='both')
    plt.legend([r'$T_x$',r'$T_y$',r'$T_z$'])
    
    plt.figure(9)
    plt.scatter(itlst,rx,s=80,facecolors='none',color='C0')
    plt.scatter(itlst,ry,s=80,facecolors='none',color='C1')
    plt.scatter(itlst,rz,s=80,facecolors='none',color='C2')
    plt.title(r'$r_{target}$ rotation vector of target over iterations')
    plt.xlabel('iteration number')
    plt.grid(b=True, which='both')
    plt.legend([r'$r_1$',r'$r_2$',r'$r_3$'])
    
    plt.figure(10)
    plt.scatter(itlst,eax,s=80,facecolors='none',color='C0')
    plt.scatter(itlst,eay,s=80,facecolors='none',color='C1')
    plt.scatter(itlst,eaz,s=80,facecolors='none',color='C2')
    plt.title(r'$r_{target}$ Euler angles of target over iterations')
    plt.xlabel('iteration number')
    plt.ylabel('degrees')
    plt.grid(b=True, which='both')
    plt.legend([r'$\theta^{euler}_x$',r'$\theta^{euler}_y$',r'$\theta^{euler}_z$'])
    
    plt.show()
