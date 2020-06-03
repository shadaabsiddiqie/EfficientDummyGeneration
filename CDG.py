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
    randRange = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
    
    layout = np.zeros((ll,lb))
    for i in range(ll):
        for j in range(lb):
            # rn = random.choice(randRange)
            # if i%5 !=0 or j%5 != 0:
            if random.random() <= probabilityDum: 
                layout[i][j] = 0
            else :
                layout[i][j] = random.choice(randRange)
                # layout[i][j] = random.random() 
    layout[userL[0]][userL[1]] = random.choice(randRange)
#-------------------------------CDG-------------------------------

def matrix2ImgCDG(userL,r,k,realPos):
    h, w = len(layout), len(layout[0])
    data = np.zeros((h, w, 3), dtype=np.uint8)
    
    #layout obstracles or none
    for h in range(len(layout)):
        for w in range(len(layout[0])):
            if layout[h][w] == 0:
                data[h,w] = [255,0,0]
            else :    
                data[h,w] = [0,255*layout[h][w],0]
    
    for p in realPos:
        data[p[0],p[1]] = [0,0,255]
    
    img = Image.fromarray(data, 'RGB')
    img = img.resize((600,600))
    img.show()

def CDG(userL, r, k):
    global layout
    realPos = []
    for x in range(k):
        (dX,dY) = polCord(userL[0], userL[1], r, x/k)
        realPos.append((dX,dY))
        # if layout[dX][dY] >=0:
        #     layout[dX][dY] = layout[dX][dY] + 1
    # matrix2ImgCDG(userL,r,k,realPos)
    return realPos 
#-------------------------------ODG-------------------------------

def realDumFromValidPos(validDumPos, k):
    noValidDum = len(validDumPos)
    realDumPos = []
    while k >= noValidDum:
        for x in validDumPos:
            realDumPos.append(x)
        k = k - noValidDum
    if k > 0: 
        lenJump = noValidDum/k 
        for x in range(k):
            realDumPos.append((validDumPos[round(x*lenJump)][0],validDumPos[round(x*lenJump)][1]))
    return realDumPos

def matrix2ImgODG(userL,r,k,realPos):
    h, w = len(layout), len(layout[0])
    data = np.zeros((h, w, 3), dtype=np.uint8)
    #layout obstracles or none
    for h in range(len(layout)):
        for w in range(len(layout[0])):
            if layout[h][w] == 0:
                data[h,w] = [255,0,0]
            else :    
                data[h,w] = [0,255*layout[h][w],0]
    
    for p in realPos:
        data[p[0],p[1]] = [255,255,255]
        # img = Image.fromarray(data, 'RGB')
        # img = img.resize((600,600))
        # img.show()


    
    img = Image.fromarray(data, 'RGB')
    img = img.resize((600,600))
    img.show()

def ODG(posDum, userL, r, k):
    global layout
    validDumPos=[]
    for x in posDum[r]:
        if layout[userL[0]+x[0]][userL[1]+x[1]] != 0:
            validDumPos.append(x)
    # return validDumPos 
    realDumPos = realDumFromValidPos(validDumPos, k)
    realPos = []
    for x in realDumPos:
        realPos.append((x[0]+userL[0],x[1]+userL[1]))
    # print(realPos)
    realPosShift = []
    for p in realPos:
        realPosShift.append((p[0]-userL[0],p[1]-userL[1]))
    
    realPosShift = sorted(realPosShift, key=sortAcos)
    realPos = []
    for p in realPosShift:
        realPos.append((p[0]+userL[0],p[1]+userL[1]))
    # print(realPos)
    # matrix2ImgODG(userL,r,k,realPos)
    return realPos
#-------------------------------EDG-------------------------------

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
            if layout[c[0]+p[0]][c[1]+p[1]] == 0:
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
            if layout[center[0]+p[0]][center[1]+p[1]] != 0:
                s.append((p[0]+center[0],p[1]+center[1]))

        ValidPosInSectors[key] = s
    return ValidPosInSectors

def matrix2ImgEDG(layout,userL,newCenters,ValidPosInSectors,realPos):
    h, w = len(layout), len(layout[0])
    data = np.zeros((h, w, 3), dtype=np.uint8)
    
    #layout obstracles or none
    for h in range(len(layout)):
        for w in range(len(layout[0])):
            if layout[h][w] == 0:
                data[h,w] = [255,0,0]
            else :    
                data[h,w] = [0,255*layout[h][w],0]
     
    #ring and sectors
    no_sectors = len(ValidPosInSectors)
    for key in ValidPosInSectors.keys():
        for p in ValidPosInSectors[key]:
            data[p[0]][p[1]] = [0,0,(255*(key+1))/no_sectors]

    #possible centers and user location
    for c in newCenters:
        data[c[0]+userL[0]][c[1]+userL[1]] = [255/2,0,255/2]
    # data[userL[0]][userL[1]] = [255,255,255]
    
    #Dummy positions
    for c in realPos:
        data[c[0]][c[1]] = [255,255,255]
    print(realPos)
    
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
    # return ValidPosInSectors
    realPos  =  []
    ran = 0
    f = 0
    # print(ValidPosInSectors)
    while len(realPos)<k:
        if  len(ValidPosInSectors[ran%k]) > 0 :
            realPos.append(ValidPosInSectors[ran%k][int(random.random()*len(ValidPosInSectors[ran%k]))])
        ran  = ran + 1
    # matrix2ImgEDG(layout,userL,newCenters,ValidPosInSectors,realPos)
    realPosShift = []
    for p in realPos:
        realPosShift.append((p[0]-bestCenters[0]-userL[0],p[1]-bestCenters[1]-userL[1]))
    
    realPosShift = sorted(realPosShift, key=sortAcos)
    realPos = []
    for p in realPosShift:
        realPos.append((p[0]+bestCenters[0]+userL[0],p[1]+bestCenters[1]+userL[1]))
    

    return (realPos,(bestCenters[0]+userL[0],bestCenters[1]+userL[1]))

#----------------------------------EDG2-----------------------------------

def normalDistribution(mu,sig,x):
    p1 = ((x-mu)/sig)
    p2 = -1*( p1**2 )/2
    e = math.exp(p2)
    return e/(sig*math.sqrt(2*math.pi))

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
            data[h,w] = [255,255,255]
            # if layout[h][w]==0:
            #     data[h,w] = [255,0,0]
            # else:
            #     data[h,w] = [0,255*layout[h][w],0]
    #ring and sectors
    # no_sectors = len(ValidPosInSectors)
    # for key in ValidPosInSectors.keys():
    #     for p in ValidPosInSectors[key]:
    #         data[p[0]][p[1]] = [0,0,(255*(key+1))/no_sectors]
    #possible centers and user location
    # for c in newCenters:
    #     data[c[0]+userL[0]][c[1]+userL[1]] = [255/2,0,255/2]
    # data[userL[0]][userL[1]] = [255,255,255]
    #Dummy positions
    # for c in edgeSectors:
    #     data[c[0]][c[1]] = [255,255,255]

    #probabulity dristributin
    for s in probabilityValidPos:
        for p in s:
            data[p[0]][p[1]] = [p[2]*30*255,p[2]*30*255,p[2]*30*255]

    img = Image.fromarray(data, 'RGB')
    img = img.resize((600,600))
    img.save('my.png')
    img.show()

    # real dummy postions
    # for p in realDummyPos:
    #     data[p[0]][p[1]] = [255,255,255]           

    # img = Image.fromarray(data, 'RGB')
    # img = img.resize((600,600))
    # img.save('my.png')
    # img.show()

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

def NormalProbDristCR(edgeSectors,ValidPosInSectors,sigCR):
    probabilityValidPos = []
    for s in ValidPosInSectors.keys():
        probTmp = []
        for p in ValidPosInSectors[s]:
            center = edgeSectors[s]
            x = math.sqrt((center[0] - p[0])**2 + (center[1] - p[1])**2)
            probTmp.append((p[0],p[1],normalDistribution(0,sigCR,x)))
        probabilityValidPos.append(probTmp)
    # print(probabilityValidPos[0])
    return probabilityValidPos

def NormalProbDristEntropy(edgeSectors,ValidPosInSectors,sigE):
    global layout
    probabilityValidPos = []
    for s in ValidPosInSectors.keys():
        probTmp = []
        for p in ValidPosInSectors[s]:
            center = edgeSectors[s]
            # x = math.sqrt((center[0] - p[0])**2 + (center[1] - p[1])**2)
            x = abs(layout[p[0]][p[1]]-layout[userL[0]][userL[1]])
            # print(x)
            probTmp.append((p[0],p[1],normalDistribution(0,sigE,x)))
        probabilityValidPos.append(probTmp)
    # print(probabilityValidPos[0])
    return probabilityValidPos

def ProbValidPosNorm(probabilityValidPos):    
    probValidPosNorm=[]
    for s in probabilityValidPos:
        normTemp = []
        sectProbSum = 0
        for p in s:
            sectProbSum = sectProbSum + p[2]
        # print(sectProbSum)
        for p in s:
            normTemp.append((p[0],p[1],p[2]/sectProbSum))
        # print(sectProbSum)
        probValidPosNorm.append(normTemp)
    # print(probValidPosNorm)
    return probValidPosNorm

def mixTwoProbabulity(normalisedProbDristCR,normalisedProbDristEntrop,a,b):
    normalisedProbDristTotal = []
    for s in range(len(normalisedProbDristCR)):
        tmp = []
        for p in range(len(normalisedProbDristCR[s])):
            # print(normalisedProbDristCR[k][2])
            x = normalisedProbDristCR[s][p][0]
            y = normalisedProbDristCR[s][p][1]
            p = a*normalisedProbDristCR[s][p][2]+b*normalisedProbDristEntrop[s][p][2]
            p = p/(a+b)
            tmp.append((x,y,p))
        normalisedProbDristTotal.append(tmp)
    return normalisedProbDristTotal

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

def EDG2(posDum, noCent, r, k, dmax, dmin, userL,sigCR,sigE):
    global layout
    newCenters = realDumFromValidPos(posDum[r], noCent)
    Udmaxmin = unionDmnx(posDum, dmax, dmin) 
    sectors = sectorCreater(Udmaxmin,k)
    bestCenters = bestCenterEDG(layout, newCenters, Udmaxmin)
    # print((bestCenters[0]+userL[0],bestCenters[1]+userL[1]) )
    ValidPosInSectors = ValidPosSectors(layout, sectors, (bestCenters[0]+userL[0],bestCenters[1]+userL[1])) 
    # return ValidPosInSectors
    edgeSectors = EdgeOfSectors(bestCenters,userL,dmax,k)
    # probabilityValidPosBinomial = ProbabilityValidPosBinomial(edgeSectors,ValidPosInSectors)
    normalProbDristCR = NormalProbDristCR(edgeSectors,ValidPosInSectors,sigCR)
    normalisedProbDristCR = ProbValidPosNorm(normalProbDristCR)
    
    # normalProbDristEntrop = NormalProbDristEntropy(edgeSectors,ValidPosInSectors,sigE)
    # normalisedProbDristEntrop = ProbValidPosNorm(normalProbDristEntrop)
    
    normalisedProbDristEntrop = normalisedProbDristCR
    # normalisedProbDristCR = normalisedProbDristEntrop

    probValidPosNorm = mixTwoProbabulity(normalisedProbDristCR,normalisedProbDristEntrop,1,0)
    realDummyPos = RealDummyPos(probValidPosNorm)
    
    realPosShift = []
    for p in realDummyPos:
        realPosShift.append((p[0]-bestCenters[0]-userL[0],p[1]-bestCenters[1]-userL[1]))
    
    realPosShift = sorted(realPosShift, key=sortAcos)
    realPos = []
    for p in realPosShift:
        realPos.append((p[0]+bestCenters[0]+userL[0],p[1]+bestCenters[1]+userL[1]))
    

    matrix2ImgEDG2(layout,userL,newCenters,ValidPosInSectors,edgeSectors,probValidPosNorm,realDummyPos)
    return (realPos , (bestCenters[0]+userL[0],bestCenters[1]+userL[1]))

#--------------------------------Results----------------------------

def areaTriangle(p1, p2, p3):
    a = math.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)
    b = math.sqrt((p2[0]-p3[0])**2+(p2[1]-p3[1])**2)
    c = math.sqrt((p3[0]-p1[0])**2+(p3[1]-p1[1])**2)
    # print(a,b,c)
    s = (a + b + c)/2
    # print(a,b,c)
    Area = math.sqrt((s*(s-a)*(s-b)*(s-c)))
    return Area

def AvgDistBetwDumm(listPoints):
    aDD = 0
    for p in range(len(listPoints)):
        p1 = listPoints[p%len(listPoints)]
        p2 = listPoints[(p+1)%len(listPoints)]
        D = math.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)
        aDD = D + aDD
    return aDD

def effectiveCR(listPoints,center):

    totalArea = 0
    for p in range(len(listPoints)):

        p1 = listPoints[p%len(listPoints)]
        p2 = listPoints[(p+1)%len(listPoints)]
        # print(p1,p2,center)
        if layout[p1[0]][p1[1]]!=0 and layout[p2[0]][p2[1]]!=0:
            totalArea = areaTriangle(p1,p2,center) + totalArea
    return totalArea

def Entropy(listPoints,layout):
    sE = 0 
    for p in listPoints:
        sE = sE + layout[p[0]][p[1]]
    total = 0
    for p in listPoints:
        pr = layout[p[0]][p[1]]/sE
        if pr != 0 :
            total = total - pr*math.log2(pr)
    return total

''' Graphes to draw
maxRadius Vs CR
Average distance between adjacent dummies based Vs k
Number of valid grids Vs the probability of the existence of obstacles
Average distance between adjacent dummies based Vs d_max
Number of dummies on obstacles based Vs k
'''

#--------------------main code------------------len(ValidPosInSectors[s])len(ValidPosInSectors[s])-

with open('maxDumPos.txt', 'rb') as handle:
  posDum = pickle.loads(handle.read())

with open('maxDumSize.txt', 'rb') as handle:
  sizeDum = pickle.loads(handle.read())

resetLayout(0.9)
# ODG(posDum, userL, 25, 10)
# EDG1-2(posDum, noCent, r, k, dmax, dmin, userL)
# dmin<r<dmax
# EDG2(posDum, 4, 3, 5, 5, 3, userL)

# -------------EDG --- Dmin Vs CR ------------------------------
# dmax = 25
# dmin = 24
# k = 10
# for dmin in range(1,dmax):
#     # print(dmin)
#     area = 0
#     noSamples = 100
#     for samp in range(noSamples):
#         resetLayout(0.3)
#         (dummyPos,center) = EDG2(posDum, 4, dmin, k, dmax, dmin, userL,2,0.001)
#         area = effectiveCR(dummyPos,center) + area
#         # area = AvgDistBetwDumm(dummyPos) + area
#         # print(area)
#     print(area/noSamples)


#--------------Average distance between adjacent dummies based Vs k----------------

# dmax = 25
# dmin = 20
# for k in range(2,41):
#     aDD = 0
#     noSamples = 100
#     for samp in range(noSamples):
#         resetLayout(0.3)
#         (dummyPos,center) = EDG(posDum, 4, dmin, k, dmax, dmin, userL)
#         # dummyPos = CDG(userL,dmax-1,k)
#         # center = userL
#         aDD = AvgDistBetwDumm(dummyPos) + aDD
#         # print(area)
#     print(aDD/noSamples)


#---------------------------------EffectiveCR Vs Obs---------------------------------

# dmax = 25
# dmin = 15
# k = 10
# for obs in range(1,10):
#     # print (obs,"---")
#     area = 0
#     noSamples = 100
#     for samp in range(noSamples):
#         resetLayout(obs/10.0)
#         # (dummyPos,center) = EDG2(posDum, 4, dmin, k, dmax, dmin, userL,2,0.001)
#         dummyPos = ODG(posDum,userL,dmax,k)
#         center = userL
#         # area = AvgDistBetwDumm(dummyPos)+area
#         area = effectiveCR(dummyPos,center) + area
#         # print(area)
#     print(area/noSamples)


#-----------------------------------Entropy Vs K-----------------------------------

# dmax = 25
# dmin = 15
# for k in range(2,41):
#     E = 0
#     noSamples = 100
#     for samp in range(noSamples):
#         resetLayout(0.3)
#         # print(layout[userL[0]][userL[1]])
#         (dummyPos,center) = EDG2(posDum, 4, dmin, k, dmax, dmin, userL,2,0.001)
#         # dummyPos = ODG(posDum,userL,dmax,k)
#         E = Entropy(dummyPos,layout) + E
#         # print(area)
#     print(E/noSamples)


#---------------------------------EffectiveCR Vs K---------------------------------
# dmax = 25
# dmin = 150.3
# for k in range(1,40):
#     area = 0
#     noSamples = 100
#     for samp in range(noSamples):
#         resetLayout(0.8)
#         # (dummyPos,center) = EDG(posDum, 4, dmin, k, dmax, dmin, userL)
#         dummyPos = ODG(posDum,userL,dmax,k)
#         center = userL
#         area = effectiveCR(dummyPos,center) + area
#         # print(area)
#     print(area/noSamples)

# -------------------------EffectiveCR Vs sigCR------------------------ gettinf best at sigCR =0 then decreasing 3,100/30
# ----------------------------Entropy Vs sigE---------------------------- 

# dmax = 25
# dmin = 15
# sigCR = 2
# sigE = 0.1 
# k = 10
# listE = []
# listCR = []
# for sigCR in range(3,50):
#     for sigE in range(1,100):
#         area = 0
#         E = 0
#         noSamples = 100
#         for samp in range(noSamples):
#             resetLayout(0.3)
#             (dummyPos,center) = EDG2(posDum, 4, dmin, k, dmax, dmin, userL,sigCR/15,sigE/500)
#             # dummyPos = ODG(posDum,userL,dmax,k)
#             # center = userL
#             # sinCR,sigE
#             area = effectiveCR(dummyPos,center) + area= 25
# dmin = 15
# k = 10
# for obs in range(1,20):
#     area = 0
#     no_VC = 0
#     noSamples = 500
#     for samp in range(noSamples):
#         resetLayout(obs/20.0)

    #     validPosition = CDG(userL,dmax,k)
    #     center = userL
    #     for p in dummyPos:
    #         if layout[p[0]][p[1]]!=0:
    #             no_VC =  no_VC + 1
    # print(no_VC/noSamples)

    #     validPosition =ODG(posDum, userL, dmax, k)
    #     center = userL
    #     for p in dummyPos:
#             E = Entropy(dummyPos,layout) + E
#             # print(area)
#         # print(area/noSamples)
#         listCR.append(area/noSamples)
#         listE.append(E/noSamples)

# print(listE)
# print(listCR)

#-----------------------------no of valied cell Vs obstracle------------------------------------
# dmax = 25
# dmin = 15
# k = 10
# for obs in range(1,20):
#     area = 0
#     no_VC = 0
#     noSamples = 500
#     for samp in range(noSamples):
#         resetLayout(obs/20.0)

    #     validPosition = CDG(userL,dmax,k)
    #     center = userL
    #     for p in dummyPos:
    #         if layout[p[0]][p[1]]!=0:
    #             no_VC =  no_VC + 1
    # print(no_VC/noSamples)

    #     validPosition =ODG(posDum, userL, dmax, k)
    #     center = userL
    #     for p in dummyPos:
    #         if layout[p[0]+userL[0]][p[1]+userL[1]]!=0:
    #             no_VC =  no_VC + 1
    # print(no_VC/noSamples)


        # validPosition = EDG(posDum, 4, dmin, k, dmax, dmin, userL)



resetLayout(0)
# (dummyPos,center) = EDG(posDum, 8, 15, 100, 25, 15, userL)
(dummyPos,center) = EDG2(posDum, 4, 15, 7, 25, 15, userL,4,0.001)
# print(effectiveCR(dummyPos,center)) 
# print(Entropy(dummyPos,layout))
# dummyPos = ODG(posDum, userL, 24, 8)

print(effectiveCR(dummyPos,userL))
# print (center)
# CDG(userL,20,7)
