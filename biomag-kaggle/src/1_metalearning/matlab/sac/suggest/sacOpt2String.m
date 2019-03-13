function cmd = sacOpt2String(OPT)
% given a option structure OPT, extract a corresponding command line str


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

if isempty(OPT)
    cmd = '';
    return;
end

cmd = Crawl(OPT);

% clean up whitespace from the string
cmd = strtrim(regexprep(cmd, '[ \t\r\f\v]+', ' '));






function optList = Crawl(OPT)

subTag = '&sub';

if ~isfield(OPT, 'explored')
    OPT(1).explored = [];
end

i = numel(OPT);
optList = {};


while i >= 1
    if isempty(OPT(i).explored)
        if OPT(i).priority > -2  % ignore any option with priority < -2
            % get the current option
            o = getOption(OPT(i).type, OPT(i).selected, OPT(i).str, OPT(i).expr);

            % mark that we have explored this option
            OPT(i).explored = 1;

            if i == 1
                % we arrived at the last option in this level of OPT
%                 if strcmpi(OPT(i).description, 'distance function')
%                             keyboard;
%                 end
%                  enc = getEncloseStr(OPT(i).enclose);
%                  optList = [o ' ' enc];
                 
                optList = o;

            else
                if isempty(OPT(i).suboptions)
                    % no sub-options to explore
%                     if strcmpi(OPT(i).description, 'distance function')
%                             keyboard;
%                     end
                        
                    subOptList = Crawl(OPT); % crawl the next option
%                     enc = getEncloseStr(OPT(i).enclose);
%                     optList = [subOptList ' ' o ' ' enc];
                    optList = [subOptList ' ' o];
                else
                    % ------- there are deeper sub-options we must explore ------

                    % enumerate the possible values this option can take
                    % TODO: be careful, this could cause problem with something
                    % that is not a List or intList

                    %v = sacGetOptionVals(OPT(i).type, OPT(i).values, OPT(i).expr);   
                    [m v] = sacGetOptionValues(OPT(i), 'max');

                    %for j = 1:length(v)
                    for j = 1:m

                        % find the value index that matches SELECTED
                        if isequal(OPT(i).selected, v{j})
                            vind = j;
                            break;
                        end

                        % we could not find a match for selected in the list of values
                        if j == length(v)
                            error('Error: selected value is not in list of possible values');
                        end
                    end

                    % crawl the matching suboption, j
                    for j = vind  
                        deepOptList = {}; %#ok<*NASGU>
                        Tsub = OPT(i).suboptions{j};

%                         if strcmpi(OPT(i).description, 'distance function')
%                             keyboard;
%                         end
                        
                        if ~isempty(Tsub)
                            % if OPT(i).suboptions{j} was not empty
                            deepOptList = Crawl(Tsub);
                            subOptList = Crawl(OPT);
%                             enc = getEncloseStr(OPT(i).enclose);
%                             optList = [subOptList ' '  o ' ' deepOptList enc];
                            optList = [subOptList ' '  o];
%                             keyboard;
                            optList = strrep(optList, subTag, deepOptList);
                            % ONLY CLOSE THE ENCLOSING! OPENING MUST BE IN STR
                        else
                            % if it was empty
                            subOptList = Crawl(OPT);
%                             enc = getEncloseStr(OPT(i).enclose);
%                             optList = [subOptList ' ' o ' ' enc];
                            optList = [subOptList ' ' o];
                            optList = strrep(optList, subTag, '');
                        end
                        %keyboard;
                    end
                end
            end

            return;
        end
    end
    i = i - 1;
end




function o = getOption(type, selected, str, expr)

switch lower(type)
    
    %case {'intlist', 'exprrange', 'realrange', 'intrange'}
    case {'intlist', 'realrange', 'intrange'}    
        o = sprintf(str, selected);
        
    case 'exprrange'
        n = selected;
        o = sprintf(str,  eval(expr));
        %keyboard;
        
    case 'list'
        o = sprintf(str,selected);
        %keyboard;
        
    case 'flag'
        if selected
            o = sprintf(str);
        else
            o = '';
        end
        
    otherwise
        error(['unknown option type: ' type]);
end


