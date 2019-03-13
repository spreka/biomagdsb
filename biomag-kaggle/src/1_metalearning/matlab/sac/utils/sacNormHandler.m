function [param normFlag] = sacNormHandler(param)
%
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


% check do see if we should do data normalization
if isempty(strfind(param, '-norm'))
    normFlag = 1;   % no normalization specified, use default normalization type
else
    normstr = regexp(param, '-norm\s*\d*', 'match');
    n = regexp(normstr{1}, '\d*', 'match');
    normFlag = str2double(strtrim(n{1}));
    param = strrep(param, normstr{1}, '');
end

