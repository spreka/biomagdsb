###################################################################################
#two modes are available:
#  in fuzzy mode, the model tries to learn the exact values for each classes
#  in discrete mode, the model learns only the class of best (max) value
fuzzy = False

#in evaluation mode only half of the train data is learned and the other half is used for evalutaion
evaluation = True

#the learned model file
modelName = "method_selector_temp1.h5"

#import plaidml.keras
#plaidml.keras.install_backend()

import keras.layers
import os.path
import skimage.io
import skimage.transform
import numpy
import image_convert
import operator

targetSize = (256, 256)
rootDir = "D:\Ábel\SZBK\Projects\Kaggle"
trainFileName = os.path.join(rootDir,"Data\\results\mergedResults.csv")
imagesDir = os.path.join(rootDir, "Data\stage1_train")

trainFile = open(trainFileName, "r")
header = trainFile.readline()

imageLines = list()
for line in trainFile:
    imageLines.append(line)

trainFile.close()

classNames = header.split(",")[1:]
print(classNames)

imagePaths = list()
imageClassValues = list()


#all images and values are loaded, because evaluation uses the same data thus all data must be present
for line in imageLines:
    splitted = line.split(",")
    imageName, _ = os.path.splitext(splitted[0])
    if(len(imageName) < 1): continue

    imagePaths.append(os.path.join(imagesDir, imageName, "images", splitted[0]))
    values = list()
    for value in splitted[1:] :
        values.append(float(value))

    imageClassValues.append(values)

imageList = list()

for imagePath in imagePaths:
    imageData = skimage.io.imread(imagePath)

    image = image_convert.convert_image(imageData, targetSize)
    if image is None: continue

    imageList.append(image)

classCount = len(classNames)

trainImagePaths = list()
trainImages = list()
trainClassValues = list()

evaluationImagePaths = list()
evaluationImages = list()
evaluationClassValues = list()

if evaluation:
    for index, image in enumerate(imageList):
        if((index % 3 == 0)):
            evaluationImagePaths.append(imagePaths[index])
            evaluationImages.append(image)
            evaluationClassValues.append(imageClassValues[index])
        else:
            trainImagePaths.append(imagePaths[index])
            trainImages.append(image)
            trainClassValues.append(imageClassValues[index])
else:
    trainImagePaths = imagePaths
    trainImages = imageList
    trainClassValues = imageClassValues
    evaluationImagePaths = imagePaths
    evaluationImages = imageList
    evaluationClassValues = imageClassValues

model = keras.models.Sequential()

model.add(keras.layers.Conv2D(32, (5, 5), input_shape = (targetSize[0], targetSize[1], 3)))
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.Dropout(0.1))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))

model.add(keras.layers.Conv2D(32, (3, 3)))
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.Dropout(0.1))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))

model.add(keras.layers.Conv2D(32, (3, 3)))
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.Dropout(0.1))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))

model.add(keras.layers.Conv2D(64, (3, 3)))
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.Dropout(0.1))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))

model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(64))
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.Dropout(0.25))
model.add(keras.layers.Dense(classCount))
model.add(keras.layers.Activation('softmax'))

if fuzzy:
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
else:
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])


if(os.path.isfile(modelName)):
    model.load_weights(modelName)
else:
    trainSize = len(trainImages)
    evalSize = len(evaluationImages)

    X_train = numpy.zeros((trainSize, targetSize[0], targetSize[1], 3))
    X_eval = numpy.zeros((evalSize, targetSize[0], targetSize[1], 3))
    Y_train = numpy.zeros((trainSize, 1))
    Y_eval = numpy.zeros((evalSize, 1))
    if fuzzy:
        Y_train.reshape(trainSize, classCount)
        Y_eval.reshape(evalSize, classCount)
        Y_train = numpy.zeros((trainSize, classCount))
        Y_eval = numpy.zeros((evalSize, classCount))

    for index, image in enumerate(trainImages) :
        X_train[index] = image
        if fuzzy: Y_train[index] = trainClassValues[index]
        else:     Y_train[index] = max(enumerate(trainClassValues[index]), key=operator.itemgetter(1))[0]

    for index, image in enumerate(evaluationImages) :
        X_eval[index] = image
        if fuzzy: Y_eval[index] = evaluationClassValues[index]
        else:     Y_eval[index] = max(enumerate(evaluationClassValues[index]), key=operator.itemgetter(1))[0]

    model.fit(X_train, Y_train, validation_data=(X_eval, Y_eval), epochs=30, batch_size=16, verbose=1)

    model.save_weights(modelName)

#evaluate

hit = 0
miss = 0
missAbs = list()
for index, image in enumerate(evaluationImages):
    images = numpy.zeros((1, image.shape[0], image.shape[1], image.shape[2]))
    images[0] = image
    originalClassIndex = max(enumerate(evaluationClassValues[index]), key=operator.itemgetter(1))[0]
    originalClass = classNames[originalClassIndex]
    prediction = model.predict(images)
    predictedClassIndex = model.predict_classes(images)[0]
    predictedClass = classNames[predictedClassIndex]
    print(os.path.basename(evaluationImagePaths[index]))
    print("  o: ", originalClass)
    print("  p: ", predictedClass)
    print("  v: ", prediction)
    if(originalClass == predictedClass):
        hit += 1
    else:
        miss += 1
        missAbs.append(abs(evaluationClassValues[index][originalClassIndex] - evaluationClassValues[index][predictedClassIndex]))

print("hits: ", hit)
print("misses: ", miss)
if miss > 0:
    missAbsArr = numpy.array(missAbs)
    print("  miss mean:    ", numpy.mean(missAbsArr))
    print("  miss std dev: ", numpy.std(missAbsArr))
    print("  miss min:     ", numpy.min(missAbsArr))
    print("  miss max:     ", numpy.max(missAbsArr))
