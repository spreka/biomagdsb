function [E s] = tempExample1()

tic;

sacInitialize;

d = sacReadArff('segment-test.arff');

% cross-validate to train a classifier and predict on the data set
[y,p] = sacCrossValidate('MultilayerPerceptron', d, 10);

% evaluate the performance of the classifier
[E s] = sacEvaluate(d.labels, p);

toc