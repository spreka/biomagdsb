function [ selimgInfo, hit, xcoord, ycoord ] = selectCellFromClassImage( x, y, mapping )
% AUTHOR:	Lassi Paavolainen
% DATE: 	April 22, 2016
% NAME: 	selectCellFromClassImage
% 
% To select a cell clickking one of those listed inside a class.
%
% INPUT:
%   x               x-coordinate of the selected cell.
%   y               y-coordinate of the selected cell.
%   mapping         Contains PlateName, ImageName, OriginalImageName and
%                   CellNumber for each cell shown.
%
% OUTPUT:
%   selimgInfo      Info of the selected image      
%   hit             0 means a wrong selection, otherwise 1.
%   xcoord          x-coordinate of the selected cell less sepsize.
%   ycoord          y-coordinate of the selected cell less sepsize.
%
% COPYRIGHT
% Advanced Cell Classifier (ACC) Toolbox. All rights reserved.
% Copyright (C) 2016 Peter Horvath,
% Synthetic and System Biology Unit, Hungarian Academia of Sciences,
% Biological Research Center, Szeged, Hungary; Institute for Molecular
% Medicine Finland, University of Helsinki, Helsinki, Finland.

imgxsize = 512;
imgysize = 512;
sepsize = 10;
cols = 7;
xcoord = x - sepsize;
ycoord = y - sepsize;

xcoord = floor(xcoord / (imgxsize+sepsize)) + 1;
ycoord = floor(ycoord / (imgysize+sepsize)) + 1;

% Check that click is over some cell image
if x >= ((xcoord-1) * imgxsize + xcoord * sepsize) && x <= ((xcoord) * imgxsize + xcoord * sepsize) && ...
   y >= ((ycoord-1) * imgysize + ycoord * sepsize) && y <= ((ycoord) * imgysize + ycoord * sepsize) && ...
   size(mapping,1) >= ycoord && size(mapping,2) >= xcoord
    selimgInfo = mapping{ycoord,xcoord};
    hit = 1;
else
    selimgInfo = {};
    hit = 0;
end
