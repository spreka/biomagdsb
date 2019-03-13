function sacWriteArff(d, filename)
% scWriteArff(data, filename)
% 
% Writes an ARFF file from a Matlab data structure.   
%
% Input:
%   filename:   The ARFF file to be written.
%   data:       A Matlab data structure containing the following fields:
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

% From the Cell Classification Library (CCL), a Matlab Toolbox for cell
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

[fid, msg] = fopen(filename, 'w');
if fid == -1
    error(msg);
end


% write comments at the top of the file
% TODO: add comments
fprintf(fid, '%%\n');

d.name = filename;

% write the database name (relation)
if ~isempty(d.name)
    fprintf(fid, ['@relation ' d.name '\n']);
end
fprintf(fid, '\n');


% write the attributes
% TODO: line up the attribute types, 'real'
N = size(d.instances,1);
M = size(d.instances,2);
for i = 1:M
    fprintf(fid, ['@attribute ' d.featureNames{i} ' REAL\n']);
end


% write the class attributes
if (~isempty(d.labels)) && (~isempty(d.classNames))
    str = '@attribute class {';
    for i = 1:length(d.classNames)-1;
        str = [str d.classNames{i} ','];
    end
    str = [str d.classNames{end} '}\n'];
    fprintf(fid, str);
end
fprintf(fid, '\n');


% write the data
fprintf(fid, ['@data\n']);

for i = 1:N
    % first the attributes
    m = [];
    for j = 1:M-1
        m = [m sprintf('%0.5g', d.instances(i,j)) ','];
    end
    m = [m sprintf('%0.5g', d.instances(i,end))];
    
    % then the class label attribute
    c = d.classNames{d.labels(i)};
    m = [m ',' c];
    fprintf(fid, [m '\n']);
end

fclose(fid);

