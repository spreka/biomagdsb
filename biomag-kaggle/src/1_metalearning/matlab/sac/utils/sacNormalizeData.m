function [data ranges] = sacNormalizeData(data, type, ranges)
% [data ranges] = sacNormalizeData(data, norm_type, ranges)
% Normalizes attributes across a sac data set.
%
% INPUT:
%  data:            a sac data structure containing a data set
%  ranges:          a matrix containing normalization parameters for each 
%                   attribute
%  norm_type:       0: No data normalization
%                   1: (Default) each attribute is normalized to have a zero mean and 1 mean/median absolute deviation (mad). 
%                   2: (Standardization) Each attribute is normalized to fall in the range [0,1].
%                   3: Each attribute is normalized to have zero mean and 1 standard deviation
%                   
% 
% OUTPUT:
%  data:            a sac data structure containing normalized data
%  ranges:          a matrix containing normalization parameters for each 
%                   attribute
%
% See also: sacPredict, sacTrain

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


dims = size(data.instances,2);

if ~exist('ranges', 'var')
    ranges = zeros(dims,2);
    givenranges = 0;
else
    if isempty(ranges)
        ranges = zeros(dims,2);
        givenranges = 0;
    else
        givenranges = 1;
    end
end

% define the default type of normalizatoin
if ~exist('type', 'var')
    type = 1;
end


%% for normalization type 0
%  do not perform any normalization
if type == 0
    ranges = [];
end

%% for normalization type 1
%  each attribute is normalized to have a zero mean and 1 mean/median average deviation
if type == 1
    %disp('   Normalizing Data: zero-mean, 1-MAD');
    for d = 1:dims
        data_d = data.instances(:,d);
    
        if givenranges == 0
            mu = mean(data_d);
            md = mad(data_d);
            ranges(d,1) = mu;
            ranges(d,2) = md;
        else
            mu = ranges(d,1);
            md = ranges(d,2);
        end
        
        % normalize the data. Handle cases where all instances have same
        % value
        if md ~= 0
            data_d = (data_d - mu) ./ (md);    
        else
            data_d = data_d - mu;
        end

        data.instances(:,d) = data_d;
        
   
    end
end



%% for normalization type 2 (standardization)
%  each attribute is normalized to fall in the range [0,1]

if type == 2
    %disp('   Normalizing Data: range of [0,1]');
    for d = 1:dims
        data_d = data.instances(:,d);

        if givenranges == 0
            mn = min(data_d);
            mx = max(data_d);
            ranges(d,1) = mn;
            ranges(d,2) = mx;
        else
            mn = ranges(d,1);
            mx = ranges(d,2);
        end

        % normalize the data. Handle cases where all instances have same
        % value
        if (mx - mn) ~= 0
            data_d = (data_d - mn) ./ (mx-mn);    
        else
            data_d = data_d - mn;
        end

        data.instances(:,d) = data_d;
    end
end

%% for normalization type 3
%  each attribute is normalized to have a zero mean and 1 standard deviation
if type == 3
    %disp('   Normalizing Data: zero-mean, 1-standard dev');
    for d = 1:dims
        data_d = data.instances(:,d);
    
        if givenranges == 0
            mu = mean(data_d);
            md = mad(data_d);
            ranges(d,1) = mu;
            ranges(d,2) = md;
        else
            mu = ranges(d,1);
            md = ranges(d,2);
        end

        % normalize the data. Handle cases where all instances have same
        % value
        if md ~= 0
            data_d = (data_d - mu) ./ (md);    
        else
            data_d = data_d - mu;
        end


        data.instances(:,d) = data_d;
        
   
    end
end

