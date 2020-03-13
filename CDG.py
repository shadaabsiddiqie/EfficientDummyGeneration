import numpy as np
import math
import pickle
import random
from PIL import Image
#baisc layout code
ll = 100 #layout length
lb = 100 #layout breath
layout = np.zeros((ll,lb))
userL = (int(ll/2),int(lb/2))
posDum = {}
sizeDum = {}

def polCord(cX, cY, r, a ):
    x = cX-r*math.sin(2*math.pi*a)
    y = cY+r*math.cos(2*math.pi*a)
    return (round(x),round(y))

def sortAcos(p): 
    if p[0]>0:
        return 360-math.degrees(math.acos( p[1]/(math.sqrt(p[1]*p[1]+p[0]*p[0]))))
    else :
        return math.degrees(math.acos( p[1]/(math.sqrt(p[1]*p[1]+p[0]*p[0]))))
    
def maxDum4r(userL,maxR,maxK):
    global layout
    # layout = np.zeros((ll,lb))
    global posDum
    for r in range(1,maxR+1):
        CDG(userL,r,maxK)
        noDum = np.cumsum(layout)[-1:][0]
        pos=[]
        for i in range(len(layout)):
            for j in range(len(layout[0])):
                if layout[i][j]==1 :
                    pos.append((i-userL[0],j-userL[1]))
        
        pos2 = sorted(pos, key=sortAcos)
        sizeDum[r]=noDum
        posDum[r]= pos2
        layout = np.zeros((ll,lb))

    with open('maxDumSize.txt', 'wb') as handle:
        pickle.dump(sizeDum, handle)
    
    with open('maxDumPos.txt', 'wb') as handle:
        pickle.dump(posDum, handle)

def resetLayout(probabilityDum):
    global layout
    layout = np.zeros((ll,lb))
    for i in range(ll):
        for j in range(lb):
            if random.random() <= probabilityDum: 
                layout[i][j] = -1

def CDG(userL, r, k):
    global layout
    for x in range(k):
        (dX,dY) = polCord(userL[0], userL[1], r, x/k)
        if layout[dX][dY] >=0:
            layout[dX][dY] = layout[dX][dY] + 1

def realDumFromValidPos(validDumPos, k):
    noValidDum = len(validDumPos)
    realDumPos = []
    while k >= noValidDum:
        for x in validDumPos:
            realDumPos.append(x)
        k = k - noValidDum
    if k > 0:
        # lenJump = int(noValidDum/k) 
        lenJump = noValidDum/k 
        for x in range(k):
            realDumPos.append((validDumPos[round(x*lenJump)][0],validDumPos[round(x*lenJump)][1]))
    return realDumPos

def ODG(posDum, userL, r, k):
    global layout
    validDumPos=[]
    # print(posDum[r])
    for x in posDum[r]:
        if layout[userL[0]+x[0]][userL[1]+x[1]] != -1:
            validDumPos.append(x)
    # print(validDumPos)
    realDumPos = realDumFromValidPos(validDumPos, k)
    # print (realDumPos)
    for x in realDumPos:
        layout[userL[0]+x[0]][userL[1]+x[1]] = layout[userL[0]+x[0]][userL[1]+x[1]] + 1     

def unionDmnx(posDum, dmax, dmin):
    Udmaxmin = []
    for r in range(dmin,dmax+1):
        for x in posDum[r]:
            if x not in Udmaxmin:
                Udmaxmin.append(x)
    return Udmaxmin

def bestCenterEDG(layout, newCenters, Umaxmin):
    bestCenter = (0,0)
    preObs = 1000000
    for c in newCenters:
        noObsC = 0
        for p in Umaxmin:
            if layout[c[0]+p[0]][c[1]+p[1]] == -1:
                noObsC = noObsC + 1
        if preObs > noObsC:
            preObs = noObsC
            bestCenter = c
    return bestCenter

def sectorCreater(Udmaxmin,noSectors):
    sectors ={}
    for no_sec in range(noSectors):
        sector = []    
        for p in Udmaxmin:
            if int((sortAcos(p)*noSectors)/360) == no_sec:
                sector.append(p)
        
        sectors[no_sec] = sector

    return sectors

def ValidPosSectors(layout, sectors, center):
    ValidPosInSectors = {}
    for key in sectors.keys():
        s = []
        for p in sectors[key]:
            if layout[center[0]+p[0]][center[1]+p[1]] != -1:
                s.append((p[0]+center[0],p[1]+center[1]))

        ValidPosInSectors[key] = s
    return ValidPosInSectors

def matrix2ImgEDG(layout,userL,newCenters,ValidPosInSectors,realPos):
    h, w = len(layout), len(layout[0])
    data = np.zeros((h, w, 3), dtype=np.uint8)
    
    #layout obstracles or none
    for h in range(len(layout)):
        for w in range(len(layout[0])):
            if layout[h][w]==0:
                data[h,w] = [0,255/2,0]
            if layout[h][w]==-1:
                data[h,w] = [255/2,0,0]
     
    
    #ring and sectors
    no_sectors = len(ValidPosInSectors)
    for key in ValidPosInSectors.keys():
        for p in ValidPosInSectors[key]:
            data[p[0]][p[1]] = [0,0,(255*(key+1))/no_sectors]

    #possible centers and user location
    for c in newCenters:
        data[c[0]+userL[0]][c[1]+userL[1]] = [255/2,0,255/2]
    data[userL[0]][userL[1]] = [255,255,255]
    
    #Dummy positions
    for c in realPos:
        data[c[0]][c[1]] = [255/1.5,255/1.5,255/1.5]
    print(realPos)
    # data[0][0]=[255,255,255]
    # data[99][99]=[0,255,0]
    
    img = Image.fromarray(data, 'RGB')
    img = img.resize((600,600))
    # img.save('my.png')
    img.show()


def EDG(posDum, noCent, r, k, dmax, dmin, userL):
    global layout
    
    newCenters = realDumFromValidPos(posDum[r], noCent)
    Udmaxmin = unionDmnx(posDum, dmax, dmin) 
    bestCenters = bestCenterEDG(layout, newCenters, Udmaxmin)
    sectors = sectorCreater(Udmaxmin,k)
    ValidPosInSectors = ValidPosSectors(layout, sectors, (bestCenters[0]+userL[0],bestCenters[1]+userL[1])) 
    
    realPos  =  []
    ran = 0
    f = 0
    # print(ValidPosInSectors)
    while len(realPos)<k:
        if  len(ValidPosInSectors[ran%k]) > 0 :
            realPos.append(ValidPosInSectors[ran%k][int(random.random()*len(ValidPosInSectors[ran%k]))])
        ran  = ran + 1
    
    # while len(realPos)<k:
    #     if  len(ValidPosInSectors[ran%k]) > 0:
    #         if ran == int(((360-sortAcos(bestCenters))*k)/360) and f == 0:
    #             realPos.append(userL)
    #             print("replaced")
    #             print(userL)
    #             f = 1
    #         else :        
    #             realPos.append(ValidPosInSectors[ran%k][int(random.random()*len(ValidPosInSectors[ran%k]))])
    #             print(ValidPosInSectors[ran%k][int(random.random()*len(ValidPosInSectors[ran%k]))])
    #         print(ran%k)
    #         print('---')
    #     ran  = ran + 1    
    matrix2ImgEDG(layout,userL,newCenters,ValidPosInSectors,realPos)

def EDG2(posDum, noCent, r, k, dmax, dmin, userL):
    global layout
    newCenters = realDumFromValidPos(posDum[r], noCent)
    Udmaxmin = unionDmnx(posDum, dmax, dmin) 
    sectors = sectorCreater(Udmaxmin,k)
    bestCenters = bestCenterEDG(layout, newCenters, Udmaxmin)
    ValidPosInSectors = ValidPosSectors(layout, sectors, (bestCenters[0]+userL[0],bestCenters[1]+userL[1])) 
    
    realPos  =  []
    ran = 0
    # print(ValidPosInSectors)
    
    while len(realPos)<k-1:
        if  len(ValidPosInSectors[ran%k]) > 0 :
            realPos.append(ValidPosInSectors[ran%k][int(random.random()*len(ValidPosInSectors[ran%k]))])
        ran  = ran + 1
    
    
    matrix2ImgEDG(layout,userL,newCenters,ValidPosInSectors,realPos)


#--------------------main code-------------------

with open('maxDumPos.txt', 'rb') as handle:
  posDum = pickle.loads(handle.read())

with open('maxDumSize.txt', 'rb') as handle:
  sizeDum = pickle.loads(handle.read())

resetLayout(0.4)
# layout[userL[0]][userL[1]]=8
# print(layout)
# ODG(posDum, userL, 2, 10)
#EDG(posDum, noCent, r, k, dmax, dmin, userL)
#dmin<r<dmax
# EDG(posDum, 4, 3, 5, 5, 3, userL)
EDG(posDum, 4, 20, 4, 21, 19, userL)
# print(layout)


