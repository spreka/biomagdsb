function [ timeInString ] = timeString(precision)
% AUTHOR:    Abel Szkalisity
% DATE:     Dec 16, 2016
% NAME:     timeString
%
% Transform time point to string.
%
% INPUT:
%   precision       An integer which defines how many 'fields' do we want
%                   to include in the final string. Fields: YYYYMMDDHHMMSS
%                   all in together at most 6.
%
% OUTPUT:
%   timeInString    A string with the current time.
%
% COPYRIGHT
% Advanced Cell Classifier (ACC) Toolbox. All rights reserved.
% Copyright (C) 2016 Peter Horvath,
% Synthetic and System Biology Unit, Hungarian Academy of Sciences,
% Biological Research Center, Szeged, Hungary; Institute for Molecular
% Medicine Finland, University of Helsinki, Helsinki, Finland.

timestamp = clock;
year = num2str(timestamp(1));
timeInString = year(3:4);
if precision>1
    timeInString = [timeInString num2str(timestamp(2),'%02d')];
    if precision>2
        timeInString = [timeInString num2str(timestamp(3),'%02d')];
        if precision>3
            timeInString = [timeInString num2str(timestamp(4),'%02d')];
            if precision>4
                timeInString = [timeInString num2str(timestamp(5),'%02d')];
                if precision>5
                    timeInString = [timeInString num2str(timestamp(6),'%02d')];
                end
            end
        end
    end
end

end

