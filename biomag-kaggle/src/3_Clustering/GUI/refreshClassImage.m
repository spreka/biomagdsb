function [] = refreshClassImage( counter, scrollpanel )
% AUTHOR:	Lassi Paavolainen, Tamas Balassa
% DATE: 	April 22, 2016
% NAME: 	refreshClassImage
% 
% To refresh the images of a class.
%
% INPUT:
%   counter             Identification number of an existing class.
%   scrollpanel         Defined using the function: 
%                       findobj(figHandle,'Tag','scrollPanel');
%
% COPYRIGHT
% Advanced Cell Classifier (ACC) Toolbox. All rights reserved.
% Copyright (C) 2016 Peter Horvath,
% Synthetic and System Biology Unit, Hungarian Academia of Sciences,
% Biological Research Center, Szeged, Hungary; Institute for Molecular
% Medicine Finland, University of Helsinki, Helsinki, Finland.

[img,~,~] = createClassImage(counter);
api = iptgetapi(scrollpanel);
api.replaceImage(img);
api.setVisibleLocation([0 0]);
