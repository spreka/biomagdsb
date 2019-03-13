function newval = sacChooseOptionValue(type, values, expr, default, weights, distribution, temperature)

% chooses a valid value for a classifier option

% temperature of 1 gives a high standard deviation. temperature of 0 gives
% a very small standard deviation.


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

switch lower(type)
    
    % ----------------- intList ----------------------
    case 'intlist'
        if numel(weights) == 1
            newval = default;
        else
            % an intlist should contain a vector of integers
            v = values;
            
            if length(weights) == length(v)
                % weighted sampling
                
                % TODO: handle all zeros in the weight list
                w = weights / sum(weights);
                newval = randsample(v,1,true,w);
            else
                error('Not enough weights for random sampling');
            end
        end
        
        
        % ----------------- List ----------------------
    case 'list'
        %         if weights == 0
        if numel(weights) == 1
            newval = default;
        else
            % List constains a cell with each possible value
            v = values;

            if length(weights) == length(v)
                % weighted sampling
                
                % TODO: handle all zeros in the weight list
                w = weights / sum(weights);
                newval = randsample(v,1,true,w);
            else
                error('Not enough weights for random sampling');    
            end
            newval = newval{1};
        end    
        
        
    % ----------------- flag ----------------------    
    case 'flag'
        if numel(weights) == 1
            newval = default;
        else
            if length(weights) == length(values)
                w = weights / sum(weights);
                vind = 1:length(values);
                newind = randsample(vind,1,true,w);
                newval = values(newind);
            else
                error('Not enough weights for random sampling');    
            end
        end

        
    % ----------------- expression range ----------------------          
    case 'exprrange'
    
        if isempty(expr)
            expr = 'n';
        end

        if weights == 0
            % if weight == 0, use the default value
            newval = default;

        else
            % otherwise, find the range of values and sample from it
            min_n = values{1};
            max_n = values{2};
            
            if strcmpi(distribution, 'uniform')
                newval = sampleUniform(min_n, max_n);
                
            elseif strcmpi(distribution, 'gaussian')
                newval = sampleGaussian(min_n, max_n, default, temperature);

            elseif strcmpi(distribution, 'rand')
                newval = sampleFlip(min_n, max_n, default, temperature);
            else
                error(['unknown distribution type: ' distribution]);
            end
            
        end

        
    % ----------------- realRange ----------------------
    case 'realrange'

        if weights == 0
            % if weight == 0, use the default value
            newval = default; %#ok<*NASGU>
        else
            min_n = values{1};
            max_n = values{2};
            
            if strcmpi(distribution, 'uniform')
                newval = sampleUniform(min_n, max_n);
                
            elseif strcmpi(distribution, 'gaussian')
                newval = sampleGaussian(min_n, max_n, default, temperature);

            elseif strcmpi(distribution, 'rand')
                newval = sampleFlip(min_n, max_n, default, temperature);
            else
                error(['unknown distribution type: ' distribution]);
            end
        end
        
    % ----------------- intRange ----------------------
    case 'intrange'

        if weights == 0
            % if weight == 0, use the default value
            newval = default; %#ok<*NASGU>
        else
            min_n = values{1};
            max_n = values{2};
            
            % TODO: randi does not work in version 2007
            %newval = randi([min_n max_n], 1, 1);
            newval = round( ((max_n - min_n) * rand(1)) + min_n);
        end   
        
    otherwise
        error(['unknown option type: ' type]);
end





function newval = sampleUniform(min_n, max_n)


s = abs(max_n - min_n);
newval = s*rand(1) + min_n;




function newval = sampleGaussian(min_n, max_n, default, temperature)


stdev = max(  [abs(default - min_n), abs(max_n - default)])*temperature;
newval = .5*stdev*randn + default;
if newval < min_n
    newval = min_n;
end
if newval > max_n
    newval = max_n;
end

                
                

function newval = sampleFlip(min_n, max_n, default, temperature)

r = rand;


if r > .5
    newval = sampleUniform(min_n, max_n);
else
    newval = sampleGaussian(min_n, max_n, default, temperature);
end
