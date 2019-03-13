function varargout = sacInit(userDefinedSACROOT)
% p = sacInit()  adds paths and returns a list of added paths
% sacInit()      adds paths
%

global SACROOT
%global logFID

% if exist('sacConfig.txt', 'file')
%     % TODO: get path to SACROOT from a config file
% end



p = cell(1,1);

if ~exist('userDefinedSACROOT', 'var')
    SACROOT = pwd;
else
    SACROOT = userDefinedSACROOT;
end;

% add path to SACROOT
p{1} = SACROOT;
if exist(p{end}, 'dir') 
    addpath(p{end});
else
    error('Could not find SaC folder');
end

% add path to classifiers
p{end+1} = [SACROOT '/classifiers'];
if exist(p{end}, 'dir') 
    addpath(p{end});
else
    error('Could not find SaC /classifiers folder');
end

% add path to classifiers/XML
p{end+1} = [SACROOT '/classifiers/xml'];
if exist(p{end}, 'dir') 
    addpath(p{end});
else
    error('Could not find SaC /classifiers/xml folder');
end

% add path to io
p{end+1} = [SACROOT '/io'];
if exist(p{end}, 'dir') 
    addpath(p{end});
else
    error('Could not find SaC /io folder');
end

% add path to utils
p{end+1} = [SACROOT '/utils'];
if exist(p{end}, 'dir') 
    addpath(p{end});
else
    error('Could not find SaC /utils folder');
end


% add path to eval
p{end+1} = [SACROOT '/eval'];
if exist(p{end}, 'dir') 
    addpath(p{end});
else
    error('Could not find SaC /eval folder');
end


% add path to suggest
p{end+1} = [SACROOT '/suggest'];
if exist(p{end}, 'dir') 
    addpath(p{end});
else
    error('Could not find SaC /suggest folder');
end


% add path to examples
p{end+1} = [SACROOT '/examples'];
if exist(p{end}, 'dir') 
    addpath(p{end});
else
    error('Could not find SaC /examples folder');
end

% add path to data
p{end+1} = [SACROOT '/data'];
if exist(p{end}, 'dir') 
    addpath(p{end});
else
    error('Could not find SaC /data folder');
end

% TODO: REMOVE IN FINAL VERSION add path to temp
%p{end+1} = [SACROOT '/temp'];
%addpath(p{end});

% % TODO: REMOVE IN FINAL VERSION add path to dev
% p{end+1} = [SACROOT '/dev'];
% addpath(p{end});

% TODO: REMOVE IN FINAL VERSION add path to backward compatibility folder
p{end+1} = [SACROOT '/utils/backwardsCompatibility'];
addpath(p{end});

% add WEKA library to the dynamic javaclasspath
WEKAROOT = [SACROOT '/thirdParty/Weka/'];
p{end+1} = WEKAROOT;
addpath(WEKAROOT);
% javaclasspathstr = [pwd '/classifiers/weka/weka.jar'];
% if isempty(ismember(javaclasspath('-dynamic'), javaclasspathstr))
%     javaclasspath([pwd '/classifiers/weka/weka.jar']);
% end
sacInitJava();
global SACROOT   %#ok<REDEF>  % necessary to recover from a javaclasspath



% add path to libsvm
LIBSVMROOT = [SACROOT '/thirdParty/LibSVM'];
p{end+1} = LIBSVMROOT;
addpath(p{end});
p{end+1} = [SACROOT '/thirdParty/LibSVM/matlab/'];
addpath(p{end});


if nargout > 0
    varargout{1} = p;
end