function b = compareVersions(V1,V2)
% AUTHOR:   Abel Szkalisity
% DATE:     June 30, 2016
% NAME:     compareVersions
%
% returns false if the second string is alphabetically smaller than the
% first one. It is true if V1 =< V2.
%
% INPUT:
%   counter     	Identification number assigned to the class.
%
% OUTPUT:
%   outimg      	Contain the table for containing the "icon images".
%   className   	Name of the class.
%   mapping     	Contains PlateName, ImageName, OriginalImageName and
%               	CellNumber for each cell shown.
%
% COPYRIGHT
% Advanced Cell Classifier (ACC) Toolbox. All rights reserved.
% Copyright (C) 2016 Peter Horvath,
% Synthetic and System Biology Unit, Hungarian Academy of Sciences,
% Biological Research Center, Szeged, Hungary; Institute for Molecular
% Medicine Finland, University of Helsinki, Helsinki, Finland.

    V1 = V1(end-7:end);
    V2 = V2(end-7:end); %cut down only the (Rxxxxl) part
    S = {V1,V2};
    S = sort(S);
    b = strcmp(S{1},V1);
end

