% optimize classifiers

close all, clear all;

% load a data set
d = sacReadArff('diabetes.arff');
%d = sacReadArff('iris.arff');

paramI = 100;
paramK = 5;
stepK = 3;
paramDepth = 5;
stepDepth = 3;

for paramK = 1:stepK:60
    for paramDepth = 1:stepDepth:60
    % partials
    classifierString = ['RandomForest -I ' num2str(paramI) ' -K ' num2str(paramK) ' -depth ' num2str(paramDepth)];
    [y,p] = sacCrossValidate(classifierString, d, 20);
    [EvalCurrent s] = sacEvaluate(d.labels, p);    
    ac(paramK, paramDepth) = EvalCurrent.ACC;

    figure(1);
    if paramK > stepK
        toShow = ac;
        toShow(toShow == 0) = NaN;
        surf(toShow(1:stepK:end, 1:stepDepth:end)); drawnow;
    end;
    end;
end;