function OPT = sacCleanOptions(OPT, fields)

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



OPT = cleanFields(OPT, fields);





function OPT = cleanFields(OPT, fields)


% clean this options fields
for j = 1:length(fields)
    if isfield(OPT, fields{j})
        OPT = rmfield(OPT, fields{j});
    end
end

%keyboard;


for i = 1:numel(OPT)
    
    if ~isempty(OPT(i).suboptions)
    	
        for j = 1:length(OPT(i).suboptions)
        
        
            Tsub = OPT(i).suboptions{j};
        
            if ~isempty(Tsub)
                OPT(i).suboptions{j} = cleanFields(Tsub, fields);
            end
        end
    
    end
    
end
