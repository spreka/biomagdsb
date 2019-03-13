function varargout = clusterSelector(varargin)
% CLUSTERSELECTOR MATLAB code for clusterSelector.fig
%      CLUSTERSELECTOR, by itself, creates a new CLUSTERSELECTOR or raises the existing
%      singleton*.
%
%      H = CLUSTERSELECTOR returns the handle to a new CLUSTERSELECTOR or the handle to
%      the existing singleton*.
%
%      CLUSTERSELECTOR('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in CLUSTERSELECTOR.M with the given input arguments.
%
%      CLUSTERSELECTOR('Property','Value',...) creates a new CLUSTERSELECTOR or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before clusterSelector_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to clusterSelector_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help clusterSelector

% Last Modified by GUIDE v2.5 08-Apr-2018 22:45:01

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @clusterSelector_OpeningFcn, ...
                   'gui_OutputFcn',  @clusterSelector_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT


% --- Executes just before clusterSelector is made visible.
function clusterSelector_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to clusterSelector (see VARARGIN)

% Choose default command line output for clusterSelector
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes clusterSelector wait for user response (see UIRESUME)
% uiwait(handles.figure1);

uData = get(handles.figure1,'UserData');
if ~isfield(uData,'clusterDir')
    %uData.clusterDir = 'E:\kaggle\Clustering\clusters\Kmeans-cosine-Best5Cluster';%'E:\kaggle\Clustering\clusters\Kmeans-cosine-Best5Cluster';    
    uData.clusterDir = uigetdir(pwd,'Select folder of clusters');    
    %uData.clusterCsv = 'E:\kaggle\Clustering\predictedStyles_Kmeans-cosine-Best5Cluster.csv';
    [f,p] = uigetfile('*.csv','Select the cluster csv file');    
    uData.clusterCsv = fullfile(p,f);        
    
    [a,b,c] = fileparts(uData.clusterCsv); 
    newCsvFile = fullfile(a,[b '_filtered' c]);
    if ~exist(newCsvFile,'file')
        copyfile(uData.clusterCsv,newCsvFile);    
    end
    uData.clusterCsv = newCsvFile;
    uData.thrash = fullfile(uData.clusterDir,'thrash');
    if ~exist(uData.thrash,'dir'), mkdir(uData.thrash); end
end
d = dir(uData.clusterDir);
dF = d(logical(cell2mat({d.isdir})));
dF = dF(3:end);

set(handles.popupmenu1,'String',{dF.name});
set(handles.figure1,'UserData',uData);


% --- Outputs from this function are returned to the command line.
function varargout = clusterSelector_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;


% --- Executes on selection change in popupmenu1.
function popupmenu1_Callback(hObject, eventdata, handles)
% hObject    handle to popupmenu1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns popupmenu1 contents as cell array
%        contents{get(hObject,'Value')} returns selected item from popupmenu1


% --- Executes during object creation, after setting all properties.
function popupmenu1_CreateFcn(hObject, eventdata, handles)
% hObject    handle to popupmenu1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in pushbutton1.
function pushbutton1_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

uData = handles.figure1.UserData;
currF = handles.popupmenu1.String{handles.popupmenu1.Value};


showTrainedCells_GUI(fullfile(uData.clusterDir,currF),uData.thrash,uData.clusterCsv,handles.popupmenu1.String);


% --- Executes on button press in pushbutton2.
function pushbutton2_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

uData = get(handles.figure1,'UserData');

clusterList = get(handles.popupmenu1,'String');
[selection,tf] = listdlg('ListString',clusterList);
if tf
    load config.mat;
    fileList = {};
    for i=1:length(selection)
        d = dir(fullfile(uData.clusterDir,clusterList{selection(i)},'*.png'));        
        fileList = [fileList; {d.name}'];
    end    
    testDir = fullfile(clusterWorkDirectory,'test');
    trainDir = fullfile(clusterWorkDirectory,'train');
    trainStyleCsv = fullfile(clusterWorkDirectory,'stylesToTrain.csv');
    predStyleCsv = uData.clusterCsv;
    extendClusterTrainingSet(testDir,trainDir,trainStyleCsv,predStyleCsv,fileList);
end

clusterSelector;
