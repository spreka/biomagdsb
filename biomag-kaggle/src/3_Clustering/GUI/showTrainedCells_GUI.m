function varargout = showTrainedCells_GUI(varargin)
% SHOWTRAINEDCELLS_GUI MATLAB code for showTrainedCells_GUI.fig
%      SHOWTRAINEDCELLS_GUI, by itself, creates a new SHOWTRAINEDCELLS_GUI or raises the existing
%      singleton*. NOTE: this figure is not singleton as it is used for
%      multiple purposes (for show similar cells and trained cells too)
%
%      H = SHOWTRAINEDCELLS_GUI returns the handle to a new SHOWTRAINEDCELLS_GUI or the handle to
%      the existing singleton*.
%
%      SHOWTRAINEDCELLS_GUI('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in SHOWTRAINEDCELLS_GUI.M with the given input arguments.
%
%      SHOWTRAINEDCELLS_GUI('Property','Value',...) creates a new SHOWTRAINEDCELLS_GUI or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before showTrainedCells_GUI_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to showTrainedCells_GUI_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help showTrainedCells_GUI

% Last Modified by GUIDE v2.5 09-Apr-2018 09:27:26

% Begin initialization code - DO NOT EDIT
gui_Singleton = 0;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @showTrainedCells_GUI_OpeningFcn, ...
                   'gui_OutputFcn',  @showTrainedCells_GUI_OutputFcn, ...
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


% --- Executes just before showTrainedCells_GUI is made visible.
function showTrainedCells_GUI_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to showTrainedCells_GUI (see VARARGIN)

% Choose default command line output for showTrainedCells_GUI
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes showTrainedCells_GUI wait for user response (see UIRESUME)
% uiwait(handles.showTrainedCellsFig);

%varargin
% counter, thrash, csvFile, allClustersNow

%counter is a string with the full path to the folder
counter = varargin{1};
uData.counter = counter;
uData.thrash = varargin{2};
uData.clusterCsv = varargin{3};
handles.classesList.String = varargin{4};

[img,className,~] = createClassImage(counter);

ax = axes('Parent',hObject);
imHandle = imshow(img,'Parent',ax);
hSP = imscrollpanel(hObject,imHandle);
set(hSP,'Units','pixels','Position',[10 70 1200 500],'Tag','scrollPanel');
api = iptgetapi(hSP);
api.setVisibleLocation([0 0]);
api.setImageButtonDownFcn({@scrollPanel_ButtonDownFcn,hObject});

%clear up previous selections        
    classImgObj.sHandle = [];
    classImgObj.selected = [];
    uData.classImgObj = classImgObj;
    set(handles.showTrainedCellsFig,'UserData',uData);
    
    set(hObject, 'Name', ['Class name: ' className]);

placeItemsOnGUI(handles);    
    



% % --- Outputs from this function are returned to the command line.
function varargout = showTrainedCells_GUI_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;


function scrollPanel_ButtonDownFcn(hObject,eventdata,fHandle)
%
%
%

%x = eventdata.IntersectionPoint(1);
%y = eventdata.IntersectionPoint(2);
% dummy code needed for old matlab
curpoint = get(gca,'CurrentPoint');
x = curpoint(1,1,1);
y = curpoint(1,2,1);
uData = get(fHandle, 'UserData');
counter = uData.counter;

[~,~,mapping] = createClassImage(counter);

classImgObj = uData.classImgObj;

[selimgInfo,hit,xcoord,ycoord] = selectCellFromClassImage(x,y,mapping);

selimgInfo.ImageName

if hit
    % Get axes object
    hSp = get(fHandle,'Children');
    axes = findobj(hSp,'Type','Axes');
    seltype = get(fHandle,'SelectionType');
    
    % Remove previous rectangle if exists, seltype == normal or counter>0
    % seltype == 'alt' for ctrl-click, 'extend' for shift-click
    if strcmp(seltype,'normal')
        if ~isempty(classImgObj.selected)
            for i=1:length(classImgObj.selected)
                if ~isempty(fieldnames(classImgObj.selected{i}))
                    delete(classImgObj.sHandle{i});
                end
            end
            classImgObj.sHandle = [];
            classImgObj.selected = [];
        end       
    end
    toDelete = 0;
    if strcmp(seltype,'alt')
        if ~isempty(classImgObj.selected)
            for i=1:length(classImgObj.selected)
                if ~isempty(fieldnames(classImgObj.selected{i})) && strcmp(classImgObj.selected{i}.ImageName,selimgInfo.ImageName)
                    delete(classImgObj.sHandle{i});
                    classImgObj.sHandle(i) = [];
                    classImgObj.selected(i) = [];
                    toDelete = 1;
                    break;
                end
            end
        end
    end
    
    if ~toDelete
        % Draw borders around the image
        if ~isempty(selimgInfo)
            lw = 3;
            imgxsize = 512;
            imgysize = 512;
            sepsize = 10;
            drawX = (xcoord-1) * imgxsize + (xcoord) * sepsize - lw;
            drawY = (ycoord-1) * imgysize + (ycoord) * sepsize - lw;
            hRec = rectangle('Position',[drawX, drawY, imgxsize + 2*lw - 1, imgysize + 2*lw - 1],'EdgeColor', 'red', 'LineWidth',lw, 'Parent', axes);
            classImgObj.sHandle{end+1} = hRec;
            classImgObj.selected{end+1} = selimgInfo;
        end    
    end
   uData.classImgObj = classImgObj;
   set(fHandle,'UserData',uData);
end


% --- Executes on button press in unclassifierButton.
function unclassifierButton_Callback(hObject, eventdata, handles)
% hObject    handle to unclassifierButton (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

figHandle = get(hObject,'Parent');
uData = get(figHandle, 'UserData');
counter = uData.counter;
classImgObj = uData.classImgObj;
thrashFolder = uData.thrash;

msgbox(' Moving images to thrash', 'Images thrasing');
if ~isempty(classImgObj.selected)
    for i=1:length(classImgObj.selected)
        if ~isempty(fieldnames(classImgObj.selected{i}))   
            %delete image
            movefile(fullfile(counter,classImgObj.selected{i}.ImageName),fullfile(thrashFolder,classImgObj.selected{i}.ImageName));
            delete(classImgObj.sHandle{i});
            %delete also from csv
            modifyEntryFromCsv(uData.clusterCsv,{classImgObj.selected{i}.ImageName},-1);
        end
    end
    classImgObj.selected = [];
    classImgObj.sHandle = [];
end

sp = findobj(figHandle,'Tag','scrollPanel');
refreshClassImage(counter, sp);


% --- Executes on button press in closeButton.
function closeButton_Callback(hObject, eventdata, handles)
% hObject    handle to closeButton (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
uData = get(handles.showTrainedCellsFig,'UserData');

counter = uData.counter;

h = msgbox(' Moving full cluster to thrash', 'Images thrasing');
d = dir(fullfile(counter,'*.png'));
for i=1:numel(d)
    movefile(fullfile(counter,d(i).name),fullfile(uData.thrash,d(i).name));    
    modifyEntryFromCsv(uData.clusterCsv,{d(i).name},-1);
end
rmdir(counter,'s')
close(h);

close;
clusterSelector;


% --- Executes on button press in jumpToCell.
function jumpToCell_Callback(hObject, eventdata, handles)
% hObject    handle to jumpToCell (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

figHandle = get(hObject,'Parent');
uData = get(figHandle, 'UserData');
classImgObj = uData.classImgObj;

info = classImgObj.selected;

if ~isempty(info) && length(info) == 1 && isfield(info{1},'ImageName')    
    img = imread(fullfile(uData.counter,info{1}.ImageName));
    figure; imshow(img);
else
    errordlg('No cell or multiple cells were selected.','Jump to cell error');
end


% --- Executes on button press in addToClass.
function addToClass_Callback(hObject, eventdata, handles)
% hObject    handle to addToClass (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

figHandle = get(hObject,'Parent');
uData = get(figHandle, 'UserData');
counter = uData.counter;
classImgObj = uData.classImgObj;

if isempty(classImgObj.selected)
    errordlg('Please, select cell or cells to be added to the class', 'Add to class error');
else
    classNameString = get(handles.classesList,'String');
    className = classNameString{get(handles.classesList,'Value')};
    if strcmp(className,'thrash')
        target = -1;
    else
        S = strsplit(className,'_');
        target = str2double(S{end});
    end
    S = strsplit(counter,filesep);
    clusterDir = strjoin(S(1:end-1),filesep);
    % Collect features and preview images of all cells        
    for i=1:length(classImgObj.selected)
        selinfo = classImgObj.selected{i};
        movefile(fullfile(counter,selinfo.ImageName),fullfile(clusterDir,className,selinfo.ImageName));
        delete(classImgObj.sHandle{i});
        modifyEntryFromCsv(uData.clusterCsv,{selinfo.ImageName},target);
    end
    classImgObj.selected = [];
    classImgObj.sHandle = [];
    
    sp = findobj(figHandle,'Tag','scrollPanel');
    refreshClassImage(counter, sp);
           
end

% --- Executes on selection change in classesList.
function classesList_Callback(hObject, eventdata, handles)
% hObject    handle to classesList (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns classesList contents as cell array
%        contents{get(hObject,'Value')} returns selected item from classesList


% --- Executes during object creation, after setting all properties.
function classesList_CreateFcn(hObject, eventdata, handles)
% hObject    handle to classesList (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in newCluster.
function newCluster_Callback(hObject, eventdata, handles)
% hObject    handle to newCluster (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

figHandle = get(hObject,'Parent');
uData = get(figHandle, 'UserData');
counter = uData.counter;
classImgObj = uData.classImgObj;

if isempty(classImgObj.selected)
    errordlg('Please, select cell or cells to be added to the class', 'Add to class error');
else
    S = strsplit(counter,filesep);
    clusterDir = strjoin(S(1:end-1),filesep);
    d = dir(clusterDir);
    maxID = 0;
    for i=1:numel(d)
        if isdir(fullfile(clusterDir,d(i).name)) && length(strsplit(d(i).name,'_'))>1
            S = strsplit(d(i).name,'_');
            base = S{1};
            nofDigits = length(S{end});
            if str2double(S{end})>maxID, maxID = str2double(S{end}); end
        end
    end    
    target = maxID + 1;
    className = [base '_' num2str(target,['%0' num2str(nofDigits) 'd'])];
    mkdir(fullfile(clusterDir,className));
    
    for i=1:length(classImgObj.selected)
        selinfo = classImgObj.selected{i};
        movefile(fullfile(counter,selinfo.ImageName),fullfile(clusterDir,className,selinfo.ImageName));
        delete(classImgObj.sHandle{i});
        modifyEntryFromCsv(uData.clusterCsv,{selinfo.ImageName},target);
    end
    classImgObj.selected = [];
    classImgObj.sHandle = [];
    
    sp = findobj(figHandle,'Tag','scrollPanel');
    refreshClassImage(counter, sp);
end

close;
clusterSelector;


% --- Executes when showTrainedCellsFig is resized.
function showTrainedCellsFig_SizeChangedFcn(hObject, eventdata, handles)
% hObject    handle to showTrainedCellsFig (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

placeItemsOnGUI(handles);

function placeItemsOnGUI(handles)
    uData = handles.showTrainedCellsFig.UserData;
    counter = uData.counter;
    xWidth = handles.showTrainedCellsFig.Position(3);
    yWidth = handles.showTrainedCellsFig.Position(4);
    sp = findobj(handles.showTrainedCellsFig,'Tag','scrollPanel');
    handles.classesList.Position(2) = 27;
    handles.addToClass.Position(2) = 27;
    handles.newCluster.Position(2) = 27;
    handles.unclassifierButton.Position(2) = 27;
    handles.jumpToCell.Position(2) = 27;
    handles.closeButton.Position(2) = 27;
    sp.Position(3:4) = [xWidth-100, yWidth-100];   

    

