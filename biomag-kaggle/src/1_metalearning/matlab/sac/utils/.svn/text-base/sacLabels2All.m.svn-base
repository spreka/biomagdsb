function [B, inds] = sacLabels2All(L, Llist)
% [B, labelList] = sacLabels2All(L, labelList)
% Converts a vector of size Nx1 containing M class labels into a binary
% matrix of size NxM.
%
% INPUT:
%  L:               A Nx1 vector containing true class labels.
%  labelList:       A 1xM cell containing a list of the possible labels. 
%
% OUTPUT:
%  B                A NxM binary matrix with 1's indicating true class
%                   labels.
%  inds:            A 1xM cell sorted according to the order. 
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


if ~exist('Llist', 'var')
    Llist = num2cell(unique(L));
end

N = numel(L);
M = numel(Llist);

B = zeros(N,M);

for i = 1:M
    B(:,i) = (L == Llist{i}); 
end

