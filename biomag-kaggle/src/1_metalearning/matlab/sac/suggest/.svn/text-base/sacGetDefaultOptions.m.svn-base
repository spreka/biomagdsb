function [cmd OPT] = sacGetDefaultOptions(classifier)
% OPT and cmd are cells containig the OPT struct and cmd string


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


% read the classifier's options from XML
xmlfile = which([classifier '.xml']);
OPT = sacReadXMLFile(xmlfile);

OPT = getDefaultOptions(OPT);
cmd = sacOpt2String(OPT);


if ~iscell(cmd)
    cmd = {cmd};
end
if ~iscell(OPT)
    OPT = {OPT};
end



function OPT = getDefaultOptions(OPT)

for i = 1:numel(OPT)
    if isempty(OPT(i).suboptions)
        % there are no sub-options
        [m default] = sacGetOptionValues(OPT(i), 'default'); %#ok<ASGLU>
        OPT(i).selected = default;
        if iscell(default)
            %error('Error: OPT(i).selected should not be a cell');
            OPT(i).selected = default{1};
        end
    else
        % ------- there are deeper sub-options we must explore ------
        
        % choose a value for this option
        [m default] = sacGetOptionValues(OPT(i), 'default'); %#ok<ASGLU>
        if iscell(default)
            %error('Error: OPT(i).selected should not be a cell');
            OPT(i).selected = default{1};
        end
        
        % find the corresponding sub-option to the chosen value
        [m v] = sacGetOptionValues(OPT(i), 'max'); %#ok<ASGLU>
        for j = 1:length(v)
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
        % assign values to the sub-options
        Tsub = OPT(i).suboptions{vind};
        if ~isempty(Tsub)
            OPT(i).suboptions{vind} = getDefaultOptions(Tsub);
        end
    end
end


