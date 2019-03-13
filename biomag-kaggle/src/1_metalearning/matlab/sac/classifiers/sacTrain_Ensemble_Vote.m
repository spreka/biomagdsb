function model = sacTrain_Ensemble_Vote(data, cls)
% classifier = sacTrain_Ensemble_Vote(data, classifierStruct)
% Trains a meta classifier using voting scheme. 
%
% INPUT:
%  data:            a sac data structure containing the training data
%  paramStruct:     a string containing options for training. Options
%                   include
%                   -norm Data normalization type
%                   -m voting scheme. 0=average probability (default),
%                   1=majority voting, 2=product of probabilities,
%                   3=maximum probability, 4=median probability
% 
%                   Default options: -m 0
% 
% OUTPUT:
%  classifier       a data structure containing a sac classifier
%
% See also: sacPredict_Ensemble_Vote, sacTrain, sacNormalizeData

% From the Suggest a Classifier Library (SAC), a Matlab Toolbox for cell
% classification in high content screening. http://www.cellclassifier.org/
% Copyright © 2011 Kevin Smith and Peter Horvath, Light Microscopy Centre 
% (LMC), Swiss Federal Institute of Technology Zurich (ETHZ), Switzerland. 
% All rights reserved.
%
% This program is free software; you can redistribute it and/or modify it 
% under the terms of the GNU General Public License version 2 (or higher) 
% as published by the Free Software Foundation. This program is 
% distributed WITHOUT ANY WARRANTY; without even the implied warranty of 
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU 
% General Public License for more details.





% handle parameter string
if ~isfield(cls, 'Options')
    parameters = '-m 0';
else
    if (isempty(cls.Options)) || strcmp(cls.Options, '')
        parameters = '-m 0';
    else
        parameters = cls.Options;
    end
end
    

% handle data normalization
[parameters normFlag] = sacNormHandler(parameters);
if normFlag
    [data dataRange] = sacNormalizeData(data, normFlag);
end


if isfield(cls, 'SubClassifiers')
    
    if normFlag
        model.dataRanges = dataRange;
    else
        model.dataRanges = [];
    end
    model.model = {};
    model.normalizeData = normFlag;
    %model.type = 'EnsembleVote';
    str = mfilename();
    model.type = str(strfind(str,'_')+1:end);
    model.options = parameters;
    
    for i = 1:length(cls.SubClassifiers)
        
        ci = cls.SubClassifiers(i);
        
        % do not normalize the data for sub-classifiers
        normstr = regexp(parameters, '-norm\s*\d*', 'match');
        if isempty(normstr)
            ci.Options = strtrim(['-norm 0' ci.Options]);
        end
        
        
        % break the input into a classifier string and parameter string
        cid = ci.ClassifierType;
       
        disp(['...training sub-classifier "' cid '"']);
    
        % create a function handle to the appropriate function
        funstr = ['sacTrain_' cid ];
        fh = str2func(funstr);
        
        % check that we have a valid classifier function
        if ~exist(funstr, 'file')
            error(['Unrecognized classifier function: ' funstr '. Check that a corresponding file exists in sacROOT/clasifiers/']);
        end
        
        % train subclassifier i
        model.SubClassifiers(i) = fh(data, ci);
        
    end
    
else
    error('No classifiers were specified to be combined.')
end

model = orderfields(model);