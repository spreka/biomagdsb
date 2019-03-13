function data = sacReadArff(filename)
% DATA = sacReadArff(filename)
% Reads data from an ARFF file into a Matlab structure.   
%
% Input:
%   filename:   The file to be loaded.
%
% Output:
%   data:       A Matlab data structure containing the data from the ARFF
%               file. The structure contains the following fields:
%
%       .name           The name (or 'relation') of the data set.
%       .featureNames   A cell containing descriptions of each feature in
%                       data set
%       .featureTypes   A cell containing data types for each feature in
%                       the data set
%       .classNames     A cell containing strings with the class names
%       .instances      An NxM matrix containing the data, where there are
%                       N data instances, each with M features
%                       (attributes).
%       .labels         An Nx1 vector containing the data labels. Might be
%                       empty in the case of unlabeled data.
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

% initialize the matlab structure
data.name = {};
data.featureNames = {};
data.featureTypes = {};
data.classNames = {};
data.instances = [];
data.labels = [];

dataflag = 0;
classposition = [];
numattributes = 0;

if ~exist(filename, 'file')
    error('Invalid filename.');
end
fid = fopen(filename);


while 1
    
    tline = fgetl(fid);
    if ~ischar(tline), break, end
    
    tline = strtrim(tline);
    
    if length(tline) > 0 %#ok<ISMT>
        
        switch tline(1)
            case '%'
                % this line is a comment, don't do anything
                continue
            case '@'
                % this line is a relation, attribute, or data label
                [token, remain] = strtok(tline);
                token = lower(token);
                
                switch token
                    case '@relation'
                        % this line gives the database name
                        remain = strrep(remain, char(39), '');
                        data.name = strtrim(remain);
                    case '@attribute'
                        [attributeToken, attributeRemain] = strtok(remain);
                        attributeToken = strrep(attributeToken, char(39), '');
                        %keyboard;
                        
                        if strcmpi(attributeToken, 'class')
                            % this line describes the class names
                            classposition = numattributes + 1;
                            remain = strtrim(remain);
                            s = regexp(remain,'[\s,{}]+','split');
                            for i = 2:length(s)
                                %c = strrep(s{i}, '}', '');
                                %c = strrep(c, '{', '');
                                c = strrep(s{i}, ',', '');
                                if ~isempty(s{i})
                                    data.classNames{end+1} = c;
                                end 
                            end
                            %keyboard;
                        else
                            % this line describes a feature attribute
                            data.featureNames{end+1} = strtrim(attributeToken);
                            % TODO: change the feature type to double,
                            % float, int, etc
                            data.featureTypes{end+1} = upper(strtrim(attributeRemain));
                            numattributes = numattributes + 1;
                        end
                        
                        
                         
                    case '@data'
                        % this line flags the start of the data
                        dataflag = 1;
                    otherwise
                       
                end
                
                
            otherwise 
                if dataflag == 1
                    % this line contains a data instance
                    % TODO: support sparse ARFF files
                    % TODO: support regression data sets
                    s = regexp(tline,'[\s,]+','split');
                    
                    if classposition <= numattributes
                        inds = 1:numattributes+1;
                        inds(classposition) = [];
                    else
                        inds = 1:numattributes;
                    end
                    
                    % features belonging to this instance
                    d = zeros(1, numattributes);
                    for i = 1:length(inds)
                        % TODO: allow for double, single, int, etc
                        d(i) = str2double(s{inds(i)});
                    end
                    data.instances(end+1,:) = d;
                    
                    % class label belonging to this instance
                    if ~isempty(classposition)
                        str = s{classposition};
                        label = find(strcmp(str, data.classNames),1);
                        data.labels(end+1,1) = label;
                    end
                else
                    disp('unknown case');
                    disp(tline);
                end
        end
    end
    
end
fclose(fid);



