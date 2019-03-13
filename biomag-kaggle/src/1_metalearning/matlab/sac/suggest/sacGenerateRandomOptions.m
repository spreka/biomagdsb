function [cmds OPTs] = sacGenerateRandomOptions(classifier, N, distribution)
%
%  Generates a list of N command strings (cmds) and option structures
%  (OPTs) randomly.
%


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



% TODO: allow options to support additional choices such as: normalization,
%       type of random distribution, feature selection, etc...


if ischar(classifier)
    % read the classifier's options from XML
    xmlfile = which([classifier '.xml']);
    OPT = sacReadXMLFile(xmlfile);
elseif isstruct(classifier)
    OPT = classifier;
else
    error('Unknown classifier input');
end


% get sampling weights for each option from the priorities
OPT = sacAssignSearchWeights(OPT);

% loop to get N randomly generated option settings
if N > 1
    cmds = cell(N,1);
    OPTs = cell(N,1);
    for i = 1:N
        OPT = sacSelectOptionRandom(OPT, distribution);  % 'uniform', 'gaussian' or 'rand'
        cmd = sacOpt2String(OPT);
        OPTs{i} = OPT;
        cmds{i} = cmd;    
    end
else
    OPTs = sacSelectOptionRandom(OPT, distribution); 
    cmds = sacOpt2String(OPTs);
end


% if cmds / OPTs contains only a single command / structure then put it
% into a cell
if ~iscell(cmds)
    cmds = {cmds};
end
if ~iscell(OPTs)
    OPTs = {OPTs};
end

