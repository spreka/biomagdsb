function [m v cmds indList] = sacGetOptionValues(OPT, varargin)
%
% [m vals] = sacGetOptionValues(OPT, 'k', k) given k, extracts appropriate values
% [m vals] = sacGetOptionValues(OPT, N)
% [m vals] = sacGetOptionValues(OPT, 'default')
% [m vals cmds] = sacGetOptionValues(OPT, N, 'includeOptionStrings')
% [m vals cmds] = sacGetOptionValues(OPT, 'includeOptionStrings', 'k', k)
% [m vals] = sacGetOptionValues(OPT, 'max')
% [m vals cmds indList] = sacGetOptionValues(OPT, N);
%
% m is the number of possible options
% val is a cell containing the possible option values
% cmds is a cell containing the strings for each possible option value
% indList points to the indexes of the returned values in the original XML list

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

optStrFlag = 0;
defaultRequestFlag = 0;
maxFlag = 0;
NMAX = 100000;


indList =[];

% handle various input arguments
for i = 1:numel(varargin)
    switch varargin{i}
        case 'default'
            defaultRequestFlag = 1;
        case 'includeOptionStrings'
            optStrFlag = 1;
        case 'max'
            maxFlag = 1;
        otherwise
            if isnumeric(varargin{i})
                if i > 1
                    if strcmpi(varargin{i-1}, 'k')
                        k = varargin{i};
                        N = max(1,round(k*sum(OPT.priority)));
                    else
                        N = varargin{i};
                    end
                else
                    N = varargin{i};
                end
            end
    end
end
if nargout > 2
    optStrFlag = 1;
end



% the default value for OPT has been requested. return it
if defaultRequestFlag
    N = 1;
end
% maximum number of values has been requested
if maxFlag
    N = NMAX;
end

%disp(['------- N = ' num2str(N) ' ---------']);

if isinf(N)
    keyboard;
end


% generate values for each possible type of option
switch lower(OPT.type)

    case 'list'
        % a List should already contain a cell where each entry is a
        % possible option value string
        v = OPT.values;
        m = numel(v);
        vin = v;
        
        % if there are more options than we requested, we will first return
        % the default option, followed by the highest priority option...
        if N < m
            [tf,defloc] = intersect(v, OPT.default);
            if isempty(tf)
                % TODO: cannot handle empty strings in lists {'', 'a', 'b'}
                error('Error: could not find default value!');
            end

            v(defloc) = [];
            if OPT.priority > 0
                priority = OPT.priority;
                priority(defloc) = [];
                [blah,inds] = sort(priority, 'descend'); %#ok<*ASGLU>
                vold = v(inds);
                v = cell(1,N);
                v{1} = OPT.default;
                if N > 1
                    v(2:N) = vold(1:N-1);
                end
            else
                v = OPT.default;
                if ~iscell(v)
%                     v = num2cell(v);
                    v = {v};
                end
            end
            
            m = numel(v);
        end

        % create a list of cmds if optStrFlag is set
        if optStrFlag
            cmds = cell(1,m);
            for j = 1:m
                cmds{j} = sprintf(OPT.str, v{j});
            end
        end
        
        % create a vector containing the list indexes if optStrFlag is set
        if optStrFlag
            indList = zeros(1,m);
            for i = 1:m
                [tf,loc] = intersect(vin, v{i});
                indList(i) = loc;
            end
        end    
        
    case 'intlist'
        % an intList should have values as a vector of integers. we first
        % return the default integer, followed my highest priority,...
        
        v = OPT.values;
        m = numel(v);
        vin = v;
        
        % if there are more options than we requested
        if N < m
            [tf,defloc] = find(v == OPT.default);
            if isempty(tf)
                error('Error: could not find default value!');
            end
            v(defloc) = [];
            if OPT.priority > 0
                priority = OPT.priority;
                priority(defloc) = [];
                [blah,inds] = sort(priority, 'descend'); %#ok<*ASGLU>
                vold = v(inds);
                v = zeros(1,N);
                v(1) = OPT.default;
                if N > 1
                    v(2:N) = vold(1:N-1);
                end
            else
                v = OPT.default;
            end
            m = numel(v);
        end
        
        v = num2cell(v);
         
        % create a list of cmds if optStrFlag is set
        if optStrFlag
            cmds = cell(1,m);
            for j = 1:m
                cmds{j} = sprintf(OPT.str, v{j});
            end
        end
        
        % create a vector containing the list indexes if optStrFlag is set
        if optStrFlag
            indList = zeros(1,m);
            if iscell(vin)                
                %vm = cell2mat(vin);
                error('vin should not be a cell!');
            else
                vm = vin;
            end
            for i = 1:m
                loc = find(vm ==  v{i}, 1, 'first');
                indList(i) = loc;
            end
        end
        
    case 'exprrange'
    
        % create n, a list of values which the expression will modify
        if N > 1
            n = OPT.values{1}: abs(OPT.values{end}-OPT.values{1})/(N-1) :OPT.values{end}; %#ok<*NASGU>
        else
            n = OPT.default;
        end
        
        % if the default value was not included in n, make n shorter and
        % add it
        [tf,defloc] = find(n == OPT.default);
        if isempty(tf)
            if N > 2
                n = OPT.values{1}: abs(OPT.values{end}-OPT.values{1})/(N-2) :OPT.values{end}; %#ok<*NASGU>
                n = unique([n OPT.default]);
            else
                n = [OPT.values{1} OPT.default];
            end
        end
        
        % apply the expression to the list n
        %v = eval(OPT.expr);
        v = n;
        v = num2cell(v);
        m = numel(v);

        % create a list of cmds if optStrFlag is set
        if optStrFlag
            cmds = cell(1,m);
            for j = 1:m
                cmds{j} = sprintf(OPT.str, v{j});
            end
        end
        
    case 'realrange'
        % will return v, a cell containing a list of possible real values
        if N > 1
            v = OPT.values{1}: abs(OPT.values{end}-OPT.values{1})/(N-1) :OPT.values{end}; %#ok<*NASGU>
        else
            v = OPT.default;
        end
        
        % if the default value was not included in v, shorten it 
        [tf,defloc] = find(v == OPT.default);
        if isempty(tf)
            if N > 2
                v = OPT.values{1}: abs(OPT.values{end}-OPT.values{1})/(N-2) :OPT.values{end}; %#ok<*NASGU>
                v = unique([v OPT.default]);
            else
                v = [OPT.values{1} OPT.default];
            end
        end
        
        % apply the expression to the list n
        v = num2cell(v);
        m = numel(v);

        % create a list of cmds if optStrFlag is set
        if optStrFlag
            cmds = cell(1,m);
            for j = 1:m
                cmds{j} = sprintf(OPT.str, v{j});
            end
        end
        
    case 'intrange'
        % will return v, a cell containing a list of possible integers
        
        % determine the maximum number of integers
        maxInts = abs(OPT.values{end} - OPT.values{1} +1);
        if N <= 1
            v = OPT.default;
        elseif N < maxInts
            v = round(OPT.values{1}: abs(OPT.values{end}-OPT.values{1})/(N-1) :OPT.values{end});
            
            % make certain the default is included
            [tf,defloc] = find(v == OPT.default);
            if isempty(tf)
                if N > 2
                    v = round(OPT.values{1}: abs(OPT.values{end}-OPT.values{1})/(N-2) :OPT.values{end});
                    v = unique([v OPT.default]);
                else
                    v = [OPT.values{1} OPT.default];
                end
            end
        else
            v = OPT.values{1}:OPT.values{end};
        end
        
        v = num2cell(v);
        m = numel(v);

        % create a list of cmds if optStrFlag is set
        if optStrFlag
            cmds = cell(1,m);
            for j = 1:m
                cmds{j} = sprintf(OPT.str, v{j});
            end
        end
        
    case 'flag'
        

        
        if N <= 1
            v = OPT.default;
        else
            v = OPT.values;
        end
        vin = v;
        v = num2cell(v);
        m = numel(v);
        
        
        % create a list of cmds if optStrFlag is set
        if optStrFlag
            cmds = cell(1,m);
            for j = 1:m
                if v{j}
                    cmds{j} = sprintf(OPT.str);
                else
                    cmds{j} = '';
                end
            end
        end
        
        
        % create a vector containing the list indexes if optStrFlag is set
        if optStrFlag
            indList = zeros(1,m);
            if iscell(vin)                
                %vm = cell2mat(vin);
                error('vin should not be a cell!');
            else
                vm = vin;
            end
            for i = 1:m
                loc = find(vm == v{i}, 1, 'first');
                indList(i) = loc;
            end
        end
        
    otherwise
        error(['Unrecognized option type: ' OPT.type]);
end




