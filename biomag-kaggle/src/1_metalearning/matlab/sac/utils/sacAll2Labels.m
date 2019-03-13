function L = sacAll2Labels(B)
% L = sacAll2Labels(L)
% Converts a class-wise matrix of size NxM (N instances, M classes) into an Nx1
% vector containing the predicted class indexes.
%
% INPUT:
%  B                A NxM matrix with values predicting the probability of
%                   the instance belonging to each class.
%
% OUTPUT:
%  L:               A Nx1 vector containing maximum class labels.
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



[dummy, L] = max(B, [] ,2); %#ok<ASGLU>
clear dummy;

