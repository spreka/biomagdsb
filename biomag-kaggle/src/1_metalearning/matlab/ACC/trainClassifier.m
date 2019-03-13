function CommonHandles = trainClassifier(CommonHandles)
% AUTHOR:   Peter Horvath, Abel Szkalisity
% DATE:     April 22, 2016
% NAME:     trainClassifier
%
% Based on settings stored in the global CommonHandles we
% train a classifier, and store it in CommonHandles again. That's why it
% does not have any input-output parameters.

%
% COPYRIGHT
% Advanced Cell Classifier (ACC) Toolbox. All rights reserved.
% Copyright (C) 2016 Peter Horvath,
% Synthetic and System Biology Unit, Hungarian Academia of Sciences,
% Biological Research Center, Szeged, Hungary; Institute for Molecular
% Medicine Finland, University of Helsinki, Helsinki, Finland.

%global CommonHandles;

% train classifier
if (CommonHandles.SALT.initialized)  
    CommonHandles.SALT.trainingData = convertACC2SALT(CommonHandles);
    if strcmp(CommonHandles.ClassifierNames{CommonHandles.SelectedClassifier},'LogitBoost_Weka') && size(CommonHandles.SALT.trainingData.instances,1)<5
        errordlg('For this classifier You need at least 5 samples!');
        uiwait(gcbf); 
    else
        CommonHandles.SALT.model = sacTrain(CommonHandles.ClassifierNames{CommonHandles.SelectedClassifier}, CommonHandles.SALT.trainingData);
    end
else
    errordlg('Please initalize SALT/sac!');
end

end