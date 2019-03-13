function OPT = sacAssignSearchWeights(OPT)
% assigns search weights to each option according to its priority and 
% number of suboptions for use in random search



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


for i = 1:numel(OPT)
    % TODO: handle all zero weights
    if isempty(OPT(i).suboptions)
        OPT(i).weight = OPT(i).priority;
        
        if numel(OPT(i).priority) == 1
            if OPT(i).priority == -1
                OPT(i).weight = 0;
            end
        end
    else
        % this option has sub-options
        if numel(OPT(i).priority) == 1
            if OPT(i).priority < 0;
                OPT(i).weight = 0;
            else
                OPT(i).weight = OPT(i).priority;
                for j = 1:length(OPT(i).suboptions)
                    Tsub = OPT(i).suboptions{j};
                    if ~isempty(Tsub)
                        Tsub = sacAssignSearchWeights(Tsub);
                    end
                    OPT(i).suboptions{j} = Tsub;
                end
            end
        else
            OPT(i).weight = OPT(i).priority;
            for j = 1:length(OPT(i).suboptions)
                Tsub = OPT(i).suboptions{j};
                if ~isempty(Tsub)
                    Tsub = sacAssignSearchWeights(Tsub);
                end
                OPT(i).suboptions{j} = Tsub;
            end
        end
    end
    if isempty(OPT(i).weight)
        error('Error: weights are empty!');
    end
end




% for i = 1:numel(OPT)
%     if isempty(OPT(i).suboptions)
%         % there are no sub-options
%         OPT(i).weight = OPT(i).priority;
%   
%         
%     else
%         % this option has sub-options
%         priority_i = OPT(i).priority;
%         w = zeros(1, length(OPT(i).suboptions));
%         
%         for j = 1:length(OPT(i).suboptions)
%             
%             Tsub = OPT(i).suboptions{j};
%             
%             if isempty(Tsub)
%                 % if a sub-option is empty, its weight is assigned to be
%                 % the priority divided by the number of search options
%                 w(j) = priority_i(i)/length(OPT(i).suboptions);
%             else
%                 % non-empty sub-option
%                 Tsub = sacAssignSearchWeights(Tsub);
%                 psub = 0;
%                 for k = 1:length(Tsub)
%                     psub = psub + Tsub(k).weight;
%                 end
%                 w(j) = psub;
%             end
%         end    
%         OPT(i).weight = w;
%     end    


%keyboard;