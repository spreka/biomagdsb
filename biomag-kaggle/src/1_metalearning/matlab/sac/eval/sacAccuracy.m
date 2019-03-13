function [ACC ERR COR INC] = sacAccuracy(y, p)
% [ACC ERR COR INC] = sacAccuracy(labels, predictions)
% Computes overall accuracy of a classifier.
%
% INPUT:
%  labels:           A vector containing true class labels.
%  predictions:      A vector containing the predicted class labels.
%
% OUTPUT:
%  ACC               Overall classification accuracy.
%  ERR               Overall classification error rate.
%  COR               Total number of correctly classified instances.
%  INC               Total number of incorrectly classified instances.
%
% See also: sacAccuracy

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

% handle the case where probabilities are given instead of predicted class
if size(p,2) > 1
    P = sacAll2Labels(p);
else 
    P = p;
end


% count the number of correct/incorrect in total
COR = sum((y == P));
INC = sum((y ~= P));

ACC = COR / (COR + INC);
ERR = 1-ACC;



