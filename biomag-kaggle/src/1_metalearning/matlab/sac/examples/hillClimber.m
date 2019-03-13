% optimize classifiers

% load a data set
d = sacReadArff('diabetes.arff');
%d = sacReadArff('iris.arff');

paramI = 100;
stepI = 1;
paramK = 5;
stepK = 1;
paramDepth = 5;
stepDepth = 1;

counter = 1;

classifierString = ['RandomForest -I ' num2str(paramI) ' -K ' num2str(paramK) ' -depth ' num2str(paramDepth)];
[y,p] = sacCrossValidate(classifierString, d, 5);
[EvalCurrentBest s] = sacEvaluate(d.labels, p);

while 1
    
    % partials
    classifierString = ['RandomForest -I ' num2str(paramI+stepI) ' -K ' num2str(paramK) ' -depth ' num2str(paramDepth)];
    [y,p] = sacCrossValidate(classifierString, d, 5);
    [EvalCurrent s] = sacEvaluate(d.labels, p);    
    delI = EvalCurrentBest.ACC - EvalCurrent.ACC;

    classifierString = ['RandomForest -I ' num2str(paramI) ' -K ' num2str(paramK+stepK) ' -depth ' num2str(paramDepth)];
    [y,p] = sacCrossValidate(classifierString, d, 5);
    [EvalCurrent s] = sacEvaluate(d.labels, p);    
    delK = EvalCurrentBest.ACC - EvalCurrent.ACC;

    classifierString = ['RandomForest -I ' num2str(paramI) ' -K ' num2str(paramK) ' -depth ' num2str(paramDepth + stepDepth)];
    [y,p] = sacCrossValidate(classifierString, d, 5);
    [EvalCurrent s] = sacEvaluate(d.labels, p);    
    delDepth = EvalCurrentBest.ACC - EvalCurrent.ACC;
    
    
    % current best
    paramI = paramI + sign(delI)*stepI;
    paramK = paramK + sign(delK)*stepK;
    paramDepth = paramDepth + sign(delDepth)*stepDepth;
    
    classifierString = ['RandomForest -I ' num2str(paramI) ' -K ' num2str(paramK) ' -depth ' num2str(paramDepth)];
    [y,p] = sacCrossValidate(classifierString, d, 5);
    [EvalCurrentBest s] = sacEvaluate(d.labels, p);
    
    accuracies(counter) = EvalCurrentBest.ACC;
    counter = counter + 1;
    figure(1);
    plot(accuracies); drawnow;
    figure(2);
    plot3(paramI, paramK, paramDepth, 'o'); hold on; drawnow;
    
end;