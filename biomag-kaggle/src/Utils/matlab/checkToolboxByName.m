function [ bool ] = checkToolboxByName( toolboxName )
% AUTHOR:   Abel Szkalisity
% DATE:     June 21, 2016
% NAME:     checkToolboxByName
%
% Check for available toolbox by name.
%
% INPUT:
%   toolboxName     the name of the searched toolbox.
%
% OUTPUT:
%   bool            0/1 if the MATLAB product is installed
%
% COPYRIGHT
% Advanced Cell Classifier (ACC) Toolbox. All rights reserved.
% Copyright (C) 2016 Peter Horvath,
% Synthetic and System Biology Unit, Hungarian Academy of Sciences,
% Biological Research Center, Szeged, Hungary; Institute for Molecular
% Medicine Finland, University of Helsinki, Helsinki, Finland.

v = ver;
bool = 0;
for i=1:length(v)
    if strcmp(v(i).Name,toolboxName)
        bool = 1;
        return;
    end
end

end

