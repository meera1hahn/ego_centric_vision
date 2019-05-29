import os
import numpy as np
import glob
import cv2
import scipy.io as sio
from skimage.morphology import skeletonize as morphThin
from multiprocessing import Pool 
from scipy.interpolate import interp1d

from bsds_utils import correspondPixels

def loadGT(gtFile):
    # load the boundary annotation into a list of numpy arrays
    data = sio.loadmat(gtFile)
    gt = data['groundTruth']
    nGt = gt.shape[1]

    gtList = []
    for i in range(0, nGt):
        curGt = gt[0,i]
        curGt = curGt['Boundaries'][0,0]
        gtList.append(curGt)

    return gtList

def readCSVFile(filename):
    lines = [line.rstrip('\n') for line in open(filename, 'r')]

    # allocate memory
    K = len(lines)
    thrs = np.zeros(K)
    cntR = np.zeros(K)
    sumR = np.zeros(K)
    cntP = np.zeros(K)
    sumP = np.zeros(K)

    # read the data
    for k in range(K):
        curRow = lines[k].split()
        thrs[k] = float(curRow[0])
        cntR[k] = float(curRow[1])
        sumR[k] = float(curRow[2])
        cntP[k] = float(curRow[3])
        sumP[k] = float(curRow[4])

    return (thrs, cntR, sumR, cntP, sumP)

def edgesEvalImg(edgeFile, gtFile, out=None, thrs=None, maxDist=None, thin=None):
    
    # default params
    if maxDist == None:
        maxDist = 0.0075
        #maxDist = 0.001875
    if thrs == None:
        thrs = 99
    if thin == None:
        thin = True

    # generate threshold list
    K = thrs
    thrs = np.linspace(1/(float(K)+1), 1-1/(float(K)+1), K)

    # read edge file
    E = cv2.imread(edgeFile, cv2.IMREAD_UNCHANGED).astype(np.float32)
    E /= 255

    # load gt file
    gtList = loadGT(gtFile)

    # allocate np array for counting
    cntR = np.zeros(K)
    sumR = np.zeros(K)
    cntP = np.zeros(K)
    sumP = np.zeros(K)

    for k in range(len(thrs)):
        E1 = (E>=thrs[k])
        # this is a similar but not exactly the same implementation 
        # of matlab's bwmorph('thin', inf) function
        if thin:
            E1 = morphThin(E1)
        E1 = E1.astype(np.float64)
        E1 = np.ascontiguousarray(E1)

        # compare to each ground truth in turn and accumulate
        # allocate memory 
        matchE = np.zeros_like(E, dtype=np.float64)
        matchG = np.zeros_like(E, dtype=np.float64)
        allG = np.zeros_like(E, dtype=np.float64)

        for g in gtList:
            G1 = g.astype(np.float64)
            G1 = np.ascontiguousarray(G1)
            #print np.max(E1)
            #print np.max(G1)
            #print maxDist
            mE1, mG1 = correspondPixels(E1, G1, maxDist)
            matchE = np.logical_or(matchE>0, mE1>0)
            matchG = matchG + (mG1>0)
            allG = allG + G1

        cntR[k] = np.sum(matchG)
        sumR[k] = np.sum(allG)
        cntP[k] = np.sum(matchE>0)
        sumP[k] = np.sum(E1>0)

    if not (out == None):
        fid = open(out, 'w')
        for k in range(len(thrs)):
            curLine = '%10g %10g %10g %10g %10g\n' % (thrs[k], cntR[k], sumR[k], cntP[k], sumP[k])
            fid.write(curLine)
        fid.close()

    return (thrs, cntR, sumR, cntP, sumP)

def edgesEvalImgPar(params):
    if 'edgeFile' in params:
        edgeFile = params['edgeFile']
    else:
        print "edgeEval: Must specify edge file"
        return
    if 'gtFile' in params:
        gtFile = params['gtFile']
    else:
        print "edgeEval: Must specify gt file"
        return

    out = None
    thrs = None
    if 'out' in params:
        out = params['out']
    if 'thrs' in params:
        thrs = params['thrs']

    thrs, cntR, sumR, cntP, sumP = edgesEvalImg(edgeFile, gtFile, out, thrs)
    return (thrs, cntR, sumR, cntP, sumP)

# compute precision, recall, f1 score
def computeRPF(cntR, sumR, cntP, sumP):
    R = cntR / (sumR + 1e-10)
    P = cntP / (sumP + 1e-10)
    F = 2*P*R / (P+R+1e-10)
    return (R, P, F)


def edgesEvalDir(edgeDir, gtDir, thrs, overwrite=None, numThreads=None):

    if numThreads == None:
        numThreads = 12

    if overwrite == None:
        overwrite = False

    # create evaluation folder if necessary
    evalDir = edgeDir + '-pyeval'
    if not os.path.exists(evalDir):
        os.makedirs(evalDir)

    # check if results exist, if so load and return
    filename = os.path.join(evalDir, 'eval_bdry_thr.txt')
    if os.path.isfile(filename) and not overwrite:
        pass

    # perform evaluation on each image
    evalParams = []
    gtList = glob.glob(os.path.join(gtDir, '*.mat'))
    for gtFile in gtList:
        imgId = os.path.basename(gtFile)
        imgId = imgId[:-4]
        edgeFile = os.path.join(edgeDir, imgId+'.png')
        resFile = os.path.join(evalDir, imgId + '_pyev.txt')
        if not os.path.isfile(resFile) or overwrite:
            # add to list
            curEvalParam = {'edgeFile':edgeFile, 'gtFile':gtFile, \
                'out':resFile, 'thrs':thrs}
            evalParams.append(curEvalParam)

    # parfor
    p = Pool(numThreads)
    p.map(edgesEvalImgPar, evalParams)
    p.close()
    p.join()

    # aggregate all results     
    K = thrs
    cntR = np.zeros(K)
    sumR = np.zeros(K)
    cntP = np.zeros(K)
    sumP = np.zeros(K)

    for gtFile in gtList:
        imgId = os.path.basename(gtFile)
        imgId = imgId[:-4]
        resFile = os.path.join(evalDir, imgId + '_pyev.txt')
        thrs1, cntR1, sumR1, cntP1, sumP1 = readCSVFile(resFile)
        thrs = thrs1
        cntR = cntR + cntR1
        sumR = sumR + sumR1
        cntP = cntP + cntP1
        sumP = sumP + sumP1

    # ODS
    P, R, F = computeRPF(cntR, sumR, cntP, sumP)
    ODS = np.max(F)

    # AP
    uR, k = np.unique(R, return_index=True)
    k = k[::-1]
    R = R[k]
    P = P[k]
    thrs = thrs[k]
    f = interp1d(R, P, kind='linear', bounds_error=False)

    AP = f(np.linspace(0,1,101))
    AP = AP[np.logical_not(np.isnan(AP)), ...]
    AP = np.sum(AP)/100

    # write to file
    fid = open(filename, 'w')
    for k in range(K):
        curLine = '%10g %10g %10g %10g\n' % (thrs[k], R[k], P[k], F[k])
        fid.write(curLine)
    fid.close()

    # print the results and return
    return (ODS, AP)
    

    