import cv2, os, shutil, time
import numpy as np
from random import shuffle
import random

trainImagesPath = "../../images/trainImages/"
trainAnnotationsPath = "../../images/trainAnnotations/"
classes = ["background", "cell_center", "cell_innerboundary", "cell_outerboundary"]
trainPatchesPath = "../../images/train_patches/"
valPatchesPath = "../../images/val_patches/"


# Extract training image patches
def extractTrainingPatches():

    for className in classes:
        if(os.path.isdir(trainPatchesPath + className) == True):
            shutil.rmtree(trainPatchesPath + className)
        os.makedirs(trainPatchesPath + className)

    numSamplesTrain = {"background": 1100, "cell_center": 1100, "cell_innerboundary": 1100, "cell_outerboundary": 1100}
    totalSamplesTrain = numSamplesTrain["background"] + numSamplesTrain["cell_center"] + numSamplesTrain["cell_innerboundary"] + numSamplesTrain["cell_outerboundary"]

    width = 1280
    height = 960
    (patchWidth, patchHeight) = (64, 64)  # window size
    (padSizeW, padSizeH) = (int(patchWidth / 2), int(patchHeight / 2))
    erosionKernel = np.ones((10, 10), np.uint8)
    dilationKernel = np.ones((30, 30), np.uint8)

    images = []
    gsImages = []
    gsErodedImages = []
    gsDilatedImages = []

    trainImageFiles = sorted(os.listdir(trainImagesPath))
    trainAnnotationFiles = sorted(os.listdir(trainAnnotationsPath))
    for ii in range(len(trainImageFiles)):
        imageFile = trainImagesPath + trainImageFiles[ii]
        annotationFile = trainAnnotationsPath + trainAnnotationFiles[ii]

        im = cv2.imread(imageFile)  # image
        gsIm = cv2.imread(annotationFile)  # gold standard image
        gsImEroded = cv2.erode(gsIm, erosionKernel, iterations=1)  # eroded gold standard image
        gsImDilated = cv2.dilate(gsIm, dilationKernel, iterations=1)  # dilated gold standard image

        images.append(im)
        gsImages.append(gsIm)
        gsErodedImages.append(gsImEroded)
        gsDilatedImages.append(gsImDilated)

    # Candidate center coordinates of image patches which are to be extracted from training image files
    imageIds = np.arange(0, len(trainImageFiles), 1)
    x = np.arange(patchWidth, width - patchWidth, 1)
    y = np.arange(patchHeight, height - patchHeight, 1)
    shuffle(imageIds)
    shuffle(x)
    shuffle(y)

    imageCount = 0
    selectedPatchCoords = []
    meanImage = np.zeros((patchHeight, patchWidth, 3))

    print("Training image patches are being saved ...")

    # Random patch extraction from an whole image for training dataset
    while (numSamplesTrain["background"] > 0 or
           numSamplesTrain["cell_center"] > 0 or
           numSamplesTrain["cell_innerboundary"] > 0 or
           numSamplesTrain["cell_outerboundary"] > 0):

        ii = random.choice(imageIds)
        xx = random.choice(x)
        yy = random.choice(y)

        # If patch coordinate is already selected, pick another random coordinate
        if ((ii, xx, yy) in selectedPatchCoords):
            continue

        selectedPatchCoords.append((ii, xx, yy))

        im = images[ii]
        gsIm = gsImages[ii]
        gsImEroded = gsErodedImages[ii]
        gsImDilated = gsDilatedImages[ii]

        top = yy - padSizeH
        bottom = yy + padSizeH
        left = xx - padSizeW
        right = xx + padSizeW
        imPatch = im[top:bottom, left:right]
        gsImPatch = gsIm[top:bottom, left:right]
        gsImErodedPatch = gsImEroded[top:bottom, left:right]
        gsImDilatedPatch = gsImDilated[top:bottom, left:right]
        centerPixelValGsEroded = gsImErodedPatch[(int)(patchHeight / 2), (int)(patchWidth / 2), 0]
        centerPixelValGsDilated = gsImDilatedPatch[(int)(patchHeight / 2), (int)(patchWidth / 2), 0]
        centerPixelValGs = gsImPatch[(int)(patchHeight / 2), (int)(patchWidth / 2), 0]

        # Classifying image patches as "background", "cell center", "cell innerboundary", and "cell outerboundary" based on their center pixel value
        if (centerPixelValGsEroded == 255 and numSamplesTrain["cell_center"] != 0):
            cv2.imwrite(trainPatchesPath + classes[1] + "/patch" + str(imageCount) + ".jpg", imPatch)
            meanImage += imPatch
            numSamplesTrain["cell_center"] -= 1
            imageCount += 1

        elif (centerPixelValGs == 255 and numSamplesTrain["cell_innerboundary"] != 0):
            cv2.imwrite(trainPatchesPath + classes[2] + "/patch" + str(imageCount) + ".jpg", imPatch)
            meanImage += imPatch
            numSamplesTrain["cell_innerboundary"] -= 1
            imageCount += 1

        elif (centerPixelValGsDilated == 255 and numSamplesTrain["cell_outerboundary"] != 0):
            cv2.imwrite(trainPatchesPath + classes[3] + "/patch" + str(imageCount) + ".jpg", imPatch)
            meanImage += imPatch
            numSamplesTrain["cell_outerboundary"] -= 1
            imageCount += 1

        elif (numSamplesTrain["background"] != 0):
            cv2.imwrite(trainPatchesPath + classes[0] + "/patch" + str(imageCount) + ".jpg", imPatch)
            meanImage += imPatch
            numSamplesTrain["background"] -= 1
            imageCount += 1

    meanImage /= totalSamplesTrain
    meanVals = np.mean(meanImage, axis=(0, 1))
    print("Training image patches are saved to a file. Mean value:", meanVals)


# Moves random number of image patches from training to validation set
def createValSet():

    for className in classes:
        if(os.path.isdir(valPatchesPath + className) == True):
            shutil.rmtree(valPatchesPath + className)
        os.makedirs(valPatchesPath + className)

        filenames = (os.listdir(trainPatchesPath+className))
        shuffle(filenames)
        filenames = filenames[0:(int)(len(filenames)/11)]

        for file in filenames:
            shutil.move(trainPatchesPath+className+"/"+file, valPatchesPath+className+"/"+file)


def createLabels(fileName, imagePath, classes):

    fileName = open(fileName, "w+")

    for ii in range(0, len(classes)):
        className = classes[ii]
        imageFiles = sorted(os.listdir(imagePath + className))
        for jj in range(0, len(imageFiles)):
            fileName.write(imagePath + className + "/" + imageFiles[jj] + " " + str(ii) + "\n")

    fileName.close()


def main():
    extractTrainingPatches()
    createValSet()

    createLabels("trainLabels.txt", trainPatchesPath, classes)
    createLabels("valLabels.txt", valPatchesPath, classes)

if __name__ == "__main__":

    start_time = time.perf_counter()

    main()

    elapsed_time = (time.perf_counter() - start_time) * 1000

    print("Elapsed time: %.3f" % elapsed_time, "ms")
