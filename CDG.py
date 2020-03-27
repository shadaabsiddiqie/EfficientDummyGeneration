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
    matrix2ImgEDG(layout,userL,newCenters,ValidPosInSectors,realPos)

def EdgeOfSectors(bestCenters,userL,dmax,k):
    edgeSectors = []
    for x in range(k):
        edgeSectors.append(polCord(bestCenters[0]+userL[0],bestCenters[1]+userL[1],dmax,(2*x+1)/(2*k)))
    return edgeSectors
                 
def matrix2ImgEDG2(layout,userL,newCenters,ValidPosInSectors,edgeSectors,probabilityValidPos,realDummyPos):
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
    for c in edgeSectors:
        data[c[0]][c[1]] = [255,255,255]

    #probabulity dristributin
    for s in probabilityValidPos:
        for p in s:
            data[p[0]][p[1]] = [0,0,p[2]*20*255]
    #real dummy postions
    for p in realDummyPos:
        data[p[0]][p[1]] = [0,255,0]           

    img = Image.fromarray(data, 'RGB')
    img = img.resize((600,600))
    # img.save('my.png')
    img.show()

def nCr(n, r): 
    return math.factorial(n)//math.factorial(r)//math.factorial(n-r)
    
def binomialDist(n,x,p):
    return nCr(n,x)*(p**x)*((1-p)**(n-x))

# def probStep(center,point,initialProb,n):
#     distX = abs(center[0]-point[0]) 
#     distY = abs(center[1]-point[1])
#     f = math.factorial
#     probX = f(n)*f(n)/(f(n-distX)*f(n+distX))
#     probY = f(n)*f(n)/(f(n-distY)*f(n+distY))
#     return initialProb*initialProb*probX*probY

def sectorWidth(edgeSectors,ValidPosInSectors):
    secWidth = []
    for s in ValidPosInSectors.keys():
        distMaxX = 0 
        distMaxY = 0
        distMax = 0 
        for p in ValidPosInSectors[s]:
            #should change n value down
            distMaxX = max(abs(edgeSectors[s][0]-p[0]),distMaxX) 
            distMaxY = max(abs(edgeSectors[s][1]-p[1]),distMaxY)
            distMax = max(distMax,max(distMaxX,distMaxY))
        secWidth.append(distMax)
    return secWidth

def ProbabilityValidPosBinomial(edgeSectors,ValidPosInSectors):
    secWidth  =  sectorWidth(edgeSectors,ValidPosInSectors)
    probabilityValidPos = []
    for s in ValidPosInSectors.keys():
        probTmp = []
        for p in ValidPosInSectors[s]:

            center = edgeSectors[s]
            point = p
            n = secWidth[s]

            distX = abs(center[0]-point[0]) 
            distY = abs(center[1]-point[1])
            f = math.factorial            
            probX = f(n)*f(n)/(f(n-distX)*f(n+distX))
            probY = f(n)*f(n)/(f(n-distY)*f(n+distY))
            probTmp.append((p[0],p[1],probX*probY))
        probabilityValidPos.append(probTmp)
    
    # print(probabilityValidPos[0])
    return probabilityValidPos

# def 

def ProbValidPosNorm(probabilityValidPos):    
    probValidPosNorm=[]
    for s in probabilityValidPos:
        normTemp = []
        sectProbSum = 0
        for p in s:
            sectProbSum = sectProbSum + p[2]
        for p in s:
            normTemp.append((p[0],p[1],p[2]/sectProbSum))
        # print(sectProbSum)
        probValidPosNorm.append(normTemp)
    # print(probValidPosNorm)
    return probValidPosNorm

def RealDummyPos(probValidPosNorm):
    realDummyPos = []
    for s in probValidPosNorm:
        ran = random.random()
        tmpSum = 0
        for p in s:
            tmpSum = tmpSum + p[2]
            if tmpSum >= ran:
                realDummyPos.append((p[0],p[1]))
                break
    return realDummyPos

def EDG2(posDum, noCent, r, k, dmax, dmin, userL):
    global layout
    newCenters = realDumFromValidPos(posDum[r], noCent)
    Udmaxmin = unionDmnx(posDum, dmax, dmin) 
    sectors = sectorCreater(Udmaxmin,k)
    bestCenters = bestCenterEDG(layout, newCenters, Udmaxmin)
    ValidPosInSectors = ValidPosSectors(layout, sectors, (bestCenters[0]+userL[0],bestCenters[1]+userL[1])) 
    edgeSectors = EdgeOfSectors(bestCenters,userL,dmax,k)
    probabilityValidPosBinomial = ProbabilityValidPosBinomial(edgeSectors,ValidPosInSectors)
    probValidPosNorm = ProbValidPosNorm(probabilityValidPosBinomial)
    realDummyPos = RealDummyPos(probValidPosNorm)
    print(realDummyPos)
    # print(probabilityValidPos)
    matrix2ImgEDG2(layout,userL,newCenters,ValidPosInSectors,edgeSectors,probValidPosNorm,realDummyPos)

    

#--------------------main code------------------len(ValidPosInSectors[s])len(ValidPosInSectors[s])-

with open('maxDumPos.txt', 'rb') as handle:
  posDum = pickle.loads(handle.read())

with open('maxDumSize.txt', 'rb') as handle:
  sizeDum = pickle.loads(handle.read())

resetLayout(0.2)
# layout[userL[0]][userL[1]]=8
# print(layout)
# ODG(posDum, userL, 2, 10)
#EDG(posDum, noCent, r, k, dmax, dmin, userL)
#dmin<r<dmax
# EDG2(posDum, 4, 3, 5, 5, 3, userL)
EDG2(posDum, 4, 20, 4, 25, 15, userL)
# print(layout)


