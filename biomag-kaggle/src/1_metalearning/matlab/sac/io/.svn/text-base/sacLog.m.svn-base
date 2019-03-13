function sacLog(priority, fid, str, varargin)


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

global displayMessagePriority
global logMessagePriority

% TODO: these globals are somehow empty
if isempty(displayMessagePriority)
    displayMessagePriority = 3;
end
if isempty(logMessagePriority)
    logMessagePriority = 5;
end


% display to the screen if the priority is sufficient
if priority <= displayMessagePriority
    fprintf(str, varargin{:});
end

% write to the log file given by "fid" if the priority is sufficient
if priority <= logMessagePriority
    fprintf(fid, str, varargin{:});
end

