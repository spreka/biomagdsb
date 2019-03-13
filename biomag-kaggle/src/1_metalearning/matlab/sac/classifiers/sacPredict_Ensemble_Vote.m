function [y,p] = sacPredict_Ensemble_Vote(classifier, data)
% [y, p] = sacPredict_Ensemble_Vote(classifier, data)
% Applies an voting ensamble of classifiers to do prediction on the data. 
%
% INPUT:
%  classifier:  a data structure containing sac classifier(s) provided
%               by sacTrain
%  data:        a sac data structure containing the training data
%
% OUTPUT:
%  y:           vector containing predicted classes for each data instance    
%  p:           matrix containing probability distribution for all classes
%  
% See also: sacTrain_Ensemble_Vote, sacPredict, sacNormalizeData

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


mode = parse_options(classifier.options);


num_classifiers = length(classifier.SubClassifiers);
num_classes = numel(data.classNames);
N = numel(data.labels);

p_list = cell(1,num_classifiers);
y_list = cell(1,num_classifiers);

for i = 1:num_classifiers
    
    ci = classifier.SubClassifiers(i);
    
    disp(['...predicting for "' ci.type '"']);
    
    % create a function handle to the appropriate function
    funstr = ['sacPredict_' ci.type ];
    fh = str2func(funstr);


    % check if the function exists
    if ~exist(funstr, 'file')
        error(['Unrecognized classifier prediction function: ' funstr '. Check that a corresponding file exists in sacROOT/clasifiers/']);
    end

    % call the prediction function
    %[y,p] = fh(ci, data);
    [y_list{i},p_list{i}] = fh(ci,data);
    
end

p = zeros(N, num_classes);

switch mode
    case 0  % average probability
        %disp('avg prob');
        
        pbig = zeros(N,num_classes,num_classifiers);
        for i = 1:num_classifiers
            pbig(:,:,i) = p_list{i};            
        end
        p = mean(pbig,3);
        
        % TODO: when there is a tie, the first class is chosen. Should it
        % be randomly chosen among the ties?
        % in case of a tie, pick max probability
        % TODO: re-calibrate probabilities
        [dummy,y] = max(p,[],2); %#ok<ASGLU>
        clear dummy;
        
    case 1  % majority voting
        %disp('maj vote');
        
        weight = 1/num_classifiers;
        row_subs = [1:N]'; %#ok<NBRAK>

        for i = 1:num_classifiers
            inds = sub2ind(size(p), row_subs, y_list{i});
            p(inds) = p(inds) + weight;
        end

        % TODO: when there is a tie, the first class is chosen. Should it
        % be randomly chosen among the ties?
        % in case of a tie, pick max probability
        [dummy,y] = max(p,[],2); %#ok<ASGLU>
        clear dummy;

     case 2  % product of probabilities
        
        pbig = zeros(N,num_classes,num_classifiers);
        for i = 1:num_classifiers
            pbig(:,:,i) = p_list{i};            
        end
        p = prod(pbig,3);
        
        % TODO: when there is a tie, the first class is chosen. Should it
        % be randomly chosen among the ties?
        % in case of a tie, pick max probability
        % TODO: recalibrate probabilities
        [dummy,y] = max(p,[],2); %#ok<ASGLU>
        clear dummy;
     
  	case 3  % maximum probability
        
        pbig = zeros(N,num_classes,num_classifiers);
        for i = 1:num_classifiers
            pbig(:,:,i) = p_list{i};            
        end
        p = max(pbig,[],3);
        
        % TODO: when there is a tie, the first class is chosen. Should it
        % be randomly chosen among the ties?
        % in case of a tie, pick max probability
        % TODO: recalibrate probabilities
        [~,y] = max(p,[],2);
        
    case 4  % median probability
        
        pbig = zeros(N,num_classes,num_classifiers);
        for i = 1:num_classifiers
            pbig(:,:,i) = p_list{i};            
        end
        p = median(pbig,3);
        
        % TODO: when there is a tie, the first class is chosen. Should it
        % be randomly chosen among the ties?
        % in case of a tie, pick max probability
        % TODO: recalibrate probabilities
        [~,y] = max(p,[],2);    
        
    otherwise
        error(['Unrecognized voting method specified: ' num2str(mode)]);
end
        







function mode = parse_options(options)


% determine the mode;
n = regexp(options, '-m\s*\d*', 'match');
n = regexp(n{1}, '\d*', 'match');
mode = str2double(strtrim(n{1}));

