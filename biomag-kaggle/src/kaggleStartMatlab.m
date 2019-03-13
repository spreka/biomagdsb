p = mfilename('fullpath');
[p,~,~] = fileparts(p);
addpath(genpath(p));