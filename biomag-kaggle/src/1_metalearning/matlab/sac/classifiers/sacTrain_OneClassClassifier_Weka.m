function classifier = sacTrain_OneClassClassifier_Weka(data, parameters)

classifierWekaType = weka.classifiers.meta.OneClassClassifier;
pa2 = '-tcl class2';

filestr = mfilename();
classifier = sacWekaTrain(classifierWekaType, data, pa2, filestr);

