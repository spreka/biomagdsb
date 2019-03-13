function varargout = vizualize_results_GUI(varargin)
% VIZUALIZE_RESULTS_GUI MATLAB code for vizualize_results_GUI.fig
%      VIZUALIZE_RESULTS_GUI, by itself, creates a new VIZUALIZE_RESULTS_GUI or raises the existing
%      singleton*.
%
%      H = VIZUALIZE_RESULTS_GUI returns the handle to a new VIZUALIZE_RESULTS_GUI or the handle to
%      the existing singleton*.
%
%      VIZUALIZE_RESULTS_GUI('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in VIZUALIZE_RESULTS_GUI.M with the given input arguments.
%
%      VIZUALIZE_RESULTS_GUI('Property','Value',...) creates a new VIZUALIZE_RESULTS_GUI or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before vizualize_results_GUI_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to vizualize_results_GUI_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help vizualize_results_GUI

% Last Modified by GUIDE v2.5 06-Apr-2018 01:03:27

% Begin initialization code - DO NOT EDIT
gui_Singleton = 0;
gui_State = struct('gui_Name',       mfilename, ...
    'gui_Singleton',  gui_Singleton, ...
    'gui_OpeningFcn', @vizualize_results_GUI_OpeningFcn, ...
    'gui_OutputFcn',  @vizualize_results_GUI_OutputFcn, ...
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


% --- Executes just before vizualize_results_GUI is made visible.
function vizualize_results_GUI_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to vizualize_results_GUI (see VARARGIN)

% Choose default command line output for vizualize_results_GUI
handles.output = hObject;
handles.globalVars = [];
handles.globalVars.visible1 = 1;
handles.globalVars.visible2 = 1;
handles.globalVars.visible3 = 1;
handles.globalVars.visible4 = 1;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes vizualize_results_GUI wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = vizualize_results_GUI_OutputFcn(hObject, eventdata, handles)
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;


% --- Executes on selection change in imageListBox.
function imageListBox_Callback(hObject, eventdata, handles)
% hObject    handle to imageListBox (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns imageListBox contents as cell array
%        contents{get(hObject,'Value')} returns selected item from imageListBox

imagesList = get(hObject, 'String');
img = imread(fullfile(get(handles.folderNameText,'String'), imagesList{get(hObject,'Value')}));

imagesc(img, 'Parent', handles.imageViewAxes);
axis image;
set(handles.imageViewAxes, 'XTick',[], 'YTick', [], 'XTickLabel',[], 'YTickLabel',[]);

handles.globalVars.selectedImageName = imagesList{get(hObject,'Value')};
handles.globalVars.selectedImage = img;
handles.globalVars.xlim = [0.5 size(img,2)+0.5];
handles.globalVars.ylim = [0.5 size(img,1)+0.5];
guidata(hObject, handles);

showResults(handles);

 if isfield(handles.globalVars,'resultsDir') && isfield(handles.globalVars,'resultsDir2') && get(handles.singleScoreCheckbox, 'Value')
     computeScore(handles);
 end

% --- Executes during object creation, after setting all properties.
function imageListBox_CreateFcn(hObject, eventdata, handles)
% hObject    handle to imageListBox (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: listbox controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in browseImageFolderButton.
function browseImageFolderButton_Callback(hObject, eventdata, handles)
% hObject    handle to browseImageFolderButton (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

if isfield(handles.globalVars,'imagesDir')
    imageDir = uigetdir (handles.globalVars.imagesDir) ;
else
    imageDir = uigetdir();
end

if imageDir
    set(handles.folderNameText, 'String', imageDir);

    imageList = dir(fullfile(imageDir,'*.png'));
    if isempty(imageList)
        imageList = dir(fullfile(imageDir,'*.tiff'));
        if isempty(imageList)
            imageList = dir(fullfile(imageDir,'*.tif'));
        end
    end
    
    set(handles.imageListBox, 'String', {imageList.name});

    handles.globalVars.selectedImageName = imageList(1).name;
    img = imread(fullfile(imageDir,imageList(1).name));
    handles.globalVars.selectedImage = img;
    handles.globalVars.xlim = [0.5 size(img,2)];
    handles.globalVars.ylim = [0.5 size(img,1)];

    handles.globalVars.imagesDir = imageDir;
    guidata(hObject,handles);

    showResults(handles);
end


% --- Executes during object creation, after setting all properties.
function imageViewAxes_CreateFcn(hObject, eventdata, handles)
% hObject    handle to imageViewAxes (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: place code in OpeningFcn to populate imageViewAxes

set(hObject, 'XTick',[], 'YTick', [], 'XTickLabel',[], 'YTickLabel',[]);


% --- Executes on selection change in resultTypeList.
function resultTypeList_Callback(hObject, eventdata, handles)
% hObject    handle to resultTypeList (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns resultTypeList contents as cell array
%        contents{get(hObject,'Value')} returns selected item from resultTypeList

handles.globalVars.resultsType = get(hObject, 'Value');
guidata(hObject, handles);


% --- Executes during object creation, after setting all properties.
function resultTypeList_CreateFcn(hObject, eventdata, handles)
% hObject    handle to resultTypeList (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

resultTypes = {'Segmentation', 'Object detection', 'Seed detection', 'Probability map'};
set(hObject, 'String', resultTypes, 'Value', 1);


% --- Executes on button press in browseResultsFolderButton.
function browseResultsFolderButton_Callback(hObject, eventdata, handles)
% hObject    handle to browseResultsFolderButton (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

if isfield(handles.globalVars, 'resultsDir')
    resultsDir = uigetdir(handles.globalVars.resultsDir);
else
    resultsDir = uigetdir();
end
if resultsDir
    handles.globalVars.resultsDir = resultsDir;
    
    fileseps = strfind(resultsDir, filesep);
    resultsDirName = resultsDir(fileseps(end)+1:end);
    parentDirName = resultsDir(fileseps(end-1)+1:fileseps(end)-1);
    set(handles.resultsFolderText, 'String',[parentDirName '/' resultsDirName],...
        'TooltipString',[parentDirName '/' resultsDirName]);

    handles.globalVars.resultsType = get(handles.resultTypeList, 'Value');
    guidata(hObject, handles);

    showResults(handles);
end





% --- Executes during object creation, after setting all properties.
function figure1_CreateFcn(hObject, eventdata, handles)
% hObject    handle to figure1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called


% --- Executes on selection change in resultTypeList2.
function resultTypeList2_Callback(hObject, eventdata, handles)
% hObject    handle to resultTypeList2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns resultTypeList2 contents as cell array
%        contents{get(hObject,'Value')} returns selected item from resultTypeList2

handles.globalVars.resultsType2 = get(hObject, 'Value');
guidata(hObject, handles);


% --- Executes during object creation, after setting all properties.
function resultTypeList2_CreateFcn(hObject, eventdata, handles)
% hObject    handle to resultTypeList2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in browseResultsFolderButton2.
function browseResultsFolderButton2_Callback(hObject, eventdata, handles)
% hObject    handle to browseResultsFolderButton2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

if isfield(handles.globalVars, 'resultsDir2')
    resultsDir = uigetdir(handles.globalVars.resultsDir2);
else
    resultsDir = uigetdir();
end
if resultsDir
    handles.globalVars.resultsDir2 = resultsDir;
    fileseps = strfind(resultsDir, filesep);
    resultsDirName = resultsDir(fileseps(end)+1:end);
    parentDirName = resultsDir(fileseps(end-1)+1:fileseps(end)-1);
    set(handles.resultsFolderText2, 'String',[parentDirName '/' resultsDirName],...
        'TooltipString',[parentDirName '/' resultsDirName]);

    handles.globalVars.resultsType2 = get(handles.resultTypeList2, 'Value');
    guidata(hObject, handles);

    showResults(handles);
end


% --- Executes on button press in ScoreButton.
function ScoreButton_Callback(hObject, eventdata, handles)
% hObject    handle to ScoreButton (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

computeScore(handles);

function computeScore(handles)
imName=handles.globalVars.selectedImageName;
[~,imBaseName,~]=fileparts(imName);

imList = dir(fullfile(handles.globalVars.resultsDir,[imBaseName '*.png']));
if isempty(imList)
    imList = dir(fullfile(handles.globalVars.resultsDir,[imBaseName '*.tiff']));
    if isempty(imList)
        imList = dir(fullfile(handles.globalVars.resultsDir,[imBaseName '*.tif']));
    end
end

imList2 = dir(fullfile(handles.globalVars.resultsDir2,[imBaseName '*.png']));
if isempty(imList2)
    imList2 = dir(fullfile(handles.globalVars.resultsDir2,[imBaseName '*.tiff']));
    if isempty(imList2)
        imList2 = dir(fullfile(handles.globalVars.resultsDir2,[imBaseName '*.tif']));
    end
end
predIm=imread(fullfile(handles.globalVars.resultsDir,imList(1).name));
gtIm=imread(fullfile(handles.globalVars.resultsDir2,imList2(1).name));

[score,gtIoUs]=evalImage(gtIm,predIm);
% set(handles.ScoreTextField,'String',['score: ' num2str(score)]);
if get(handles.singleScoreCheckbox, 'Value')
    centers = regionprops(gtIm,'Centroid');
    for i=1:numel(centers)
        if gtIoUs(i)<0.5
            text(centers(i).Centroid(1),centers(i).Centroid(2),num2str(gtIoUs(i), '%0.3f'),'Parent',handles.imageViewAxes,'Color','c','EdgeColor',[0 1 1]);
        else
            text(centers(i).Centroid(1),centers(i).Centroid(2),num2str(gtIoUs(i), '%0.3f'),'Parent',handles.imageViewAxes,'Color','c');
        end
    end
end
% guidata(hObject,handles);


% --- Executes on selection change in resultTypeList3.
function resultTypeList3_Callback(hObject, eventdata, handles)
% hObject    handle to resultTypeList3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns resultTypeList3 contents as cell array
%        contents{get(hObject,'Value')} returns selected item from resultTypeList3

handles.globalVars.resultsType3 = get(hObject, 'Value');
guidata(hObject, handles);


% --- Executes during object creation, after setting all properties.
function resultTypeList3_CreateFcn(hObject, eventdata, handles)
% hObject    handle to resultTypeList3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in browseResultsFolderButton3.
function browseResultsFolderButton3_Callback(hObject, eventdata, handles)
% hObject    handle to browseResultsFolderButton3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


if isfield(handles.globalVars, 'resultsDir3')
    resultsDir = uigetdir(handles.globalVars.resultsDir3);
else
    resultsDir = uigetdir();
end
if resultsDir
    handles.globalVars.resultsDir3 = resultsDir;
    
    fileseps = strfind(resultsDir, filesep);
    resultsDirName = resultsDir(fileseps(end)+1:end);
    parentDirName = resultsDir(fileseps(end-1)+1:fileseps(end)-1);
    set(handles.resultsFolderText3, 'String',[parentDirName '/' resultsDirName],...
        'TooltipString',[parentDirName '/' resultsDirName]);

    handles.globalVars.resultsType3 = get(handles.resultTypeList3, 'Value');
    guidata(hObject, handles);

    showResults(handles);
end


% --- Executes on selection change in resultTypeList4.
function resultTypeList4_Callback(hObject, eventdata, handles)
% hObject    handle to resultTypeList4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns resultTypeList4 contents as cell array
%        contents{get(hObject,'Value')} returns selected item from resultTypeList4

handles.globalVars.resultsType4 = get(hObject, 'Value');
guidata(hObject, handles);


% --- Executes during object creation, after setting all properties.
function resultTypeList4_CreateFcn(hObject, eventdata, handles)
% hObject    handle to resultTypeList4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in browseResultsFolderButton4.
function browseResultsFolderButton4_Callback(hObject, eventdata, handles)
% hObject    handle to browseResultsFolderButton4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


if isfield(handles.globalVars, 'resultsDir4')
    resultsDir = uigetdir(handles.globalVars.resultsDir4);
else
    resultsDir = uigetdir();
end
if resultsDir
    handles.globalVars.resultsDir4 = resultsDir;
    
    fileseps = strfind(resultsDir, filesep);
    resultsDirName = resultsDir(fileseps(end)+1:end);
    parentDirName = resultsDir(fileseps(end-1)+1:fileseps(end)-1);
    set(handles.resultsFolderText4, 'String',[parentDirName '/' resultsDirName],...
        'TooltipString',[parentDirName '/' resultsDirName]);

    handles.globalVars.resultsType4 = get(handles.resultTypeList4, 'Value');
    guidata(hObject, handles);

    showResults(handles);
end



% --- Executes on mouse press over axes background.
function imageViewAxes_ButtonDownFcn(hObject, eventdata, handles)
% hObject    handle to imageViewAxes (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes on mouse press over figure background.
function figure1_ButtonDownFcn(hObject, eventdata, handles)
% hObject    handle to figure1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% --- Executes on button press in singleScoreCheckbox.
function singleScoreCheckbox_Callback(hObject, eventdata, handles)
% hObject    handle to singleScoreCheckbox (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of singleScoreCheckbox


% --- Executes on button press in visible1.
function visible1_Callback(hObject, eventdata, handles)
% hObject    handle to visible1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of visible1

handles.globalVars.visible1 = get(hObject,'Value');
guidata(hObject, handles);
showResults(handles);


% --- Executes on button press in visible2.
function visible2_Callback(hObject, eventdata, handles)
% hObject    handle to visible2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of visible2

handles.globalVars.visible2 = get(hObject,'Value');
guidata(hObject, handles);
showResults(handles);


% --- Executes on button press in visible3.
function visible3_Callback(hObject, eventdata, handles)
% hObject    handle to visible3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of visible3

handles.globalVars.visible3 = get(hObject,'Value');
guidata(hObject, handles);
showResults(handles);


% --- Executes on button press in visible4.
function visible4_Callback(hObject, eventdata, handles)
% hObject    handle to visible4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of visible4

handles.globalVars.visible4 = get(hObject,'Value');
guidata(hObject, handles);
showResults(handles);


% ================ additional functions ========================


function showResults(handles)

if isfield(handles, 'globalVars')
    if isfield(handles.globalVars, 'imagesDir')
        if isfield(handles.globalVars, 'selectedImage')
            cla(handles.imageViewAxes);
            imagesc(handles.globalVars.selectedImage, 'Parent', handles.imageViewAxes);
            axis image;
            set(handles.imageViewAxes, 'XLim', handles.globalVars.xlim);
            set(handles.imageViewAxes, 'YLim', handles.globalVars.ylim);
            
            set(handles.imageViewAxes, 'XTick',[], 'YTick', [], 'XTickLabel',[], 'YTickLabel',[]);
            selectedImageName = handles.globalVars.selectedImageName;
        end
    end
    if isfield(handles.globalVars,'resultsType') && handles.globalVars.visible1
        resultsType = handles.globalVars.resultsType;        
        putAnnotation(handles,resultsType,handles.globalVars.resultsDir,selectedImageName,[1 0 0],1);
    end
    if isfield(handles.globalVars,'resultsType2') && handles.globalVars.visible2
        resultsType = handles.globalVars.resultsType2;
        putAnnotation(handles,resultsType,handles.globalVars.resultsDir2,selectedImageName,[0 1 0],2);
    end
    if isfield(handles.globalVars,'resultsType3') && handles.globalVars.visible3
        resultsType = handles.globalVars.resultsType3;        
        putAnnotation(handles,resultsType,handles.globalVars.resultsDir3,selectedImageName,[1 0 1],3);
    end
    if isfield(handles.globalVars,'resultsType4') && handles.globalVars.visible4
        resultsType = handles.globalVars.resultsType4;
        putAnnotation(handles,resultsType,handles.globalVars.resultsDir4,selectedImageName,[0 1 1],4);
    end
else
end

function putAnnotation(handles,resultsType,resultsDir,selectedImageName,color,position)
    switch resultsType
            case 1 % segmentation
                imageName = selectedImageName;
                [~,imageBaseName,~] = fileparts(imageName);
                segmList = dir(fullfile(resultsDir,[imageBaseName '*.png']));
                if isempty(segmList)
                    segmList = dir(fullfile(resultsDir,[imageBaseName '*.tiff']));
                    if isempty(segmList)
                        segmList = dir(fullfile(resultsDir,[imageBaseName '*.tif']));
                    end
                end
                if ~isempty(segmList)
                    segmentation = imread(fullfile(resultsDir,segmList(1).name));
                    perim = labelperim(segmentation)>0;
%                     perim = bwlabelperim(segmentation);
                    perimImage = cat(3, 255*uint8(perim>0)*color(1), 255*uint8(perim>0)*color(2), 255*uint8(perim>0)*color(3));
                   
                    axes(handles.imageViewAxes); hold on;
                    hp = imagesc(perimImage, 'Parent', handles.imageViewAxes);
                    set(hp,'AlphaData', perim>0);
                    
                    props = regionprops(segmentation, 'EquivDiameter');
                    props(cat(1,props.EquivDiameter)==0) = [];
                    medianArea = median(cat(1,props.EquivDiameter));
                    stdArea = std(cat(1,props.EquivDiameter));
                    
                    ht = text(size(segmentation,2)-60, 20+(position-1)*50, sprintf('%0.0f+/-%0.2f',medianArea,stdArea), 'FontSize', 16, 'Color', color); %, 'BackgroundColor', 1-color);
                    bboxHt = get(ht,'Extent');
                    corners = [bboxHt(1)-2 bboxHt(2); bboxHt(1)+bboxHt(3)+2 bboxHt(2); bboxHt(1)+bboxHt(3)+2 bboxHt(2)-bboxHt(4); bboxHt(1)-2 bboxHt(2)-bboxHt(4)];
                    p = patch(corners(:,1), corners(:,2), 1-color);
                    set(p,'FaceAlpha',0.25, 'EdgeColor', color);
                else
                    %warndlg(sprintf('No result for image %s is available.',imageBaseName));
                end
                
                if exist(fullfile(resultsDir, 'scores.csv'), 'file')
                    scoreData = importdata(fullfile(resultsDir, 'scores.csv'));
                    imageIdx = find( cellfun(@(x) strcmp(segmList(1).name,x), scoreData.textdata) );
                    segmScore = scoreData.data(imageIdx(1));
                    ht = text(10, 20+(position-1)*50, num2str(segmScore,'%0.4f'), 'FontSize', 16, 'Color', color); %, 'BackgroundColor', 1-color);
                    bboxHt = get(ht,'Extent');
                    corners = [bboxHt(1)-2 bboxHt(2); bboxHt(1)+bboxHt(3)+2 bboxHt(2); bboxHt(1)+bboxHt(3)+2 bboxHt(2)-bboxHt(4); bboxHt(1)-2 bboxHt(2)-bboxHt(4)];
                    p = patch(corners(:,1), corners(:,2), 1-color);
                    set(p,'FaceAlpha',0.25, 'EdgeColor', color);
                end
                
            case 2 % object detection
                imageName = selectedImageName;
                [~,imageBaseName,~] = fileparts(imageName);
                bboxList = dir(fullfile(resultsDir,[imageBaseName '*.csv']));
                if isempty(bboxList)
                    bboxList = dir(fullfile(resultsDir,[imageBaseName '*.txt']));
                end
                if ~isempty(bboxList)
                    bboxes = readBoundingBoxes(fullfile(resultsDir,bboxList(1).name));
                    if size(bboxes,2)>4
                        scores = bboxes(:,5);
                    else
                        scores = ones(size(bboxes,1),1);
                    end
                    for i=1:size(bboxes,1)
                        rectangle('Position', [bboxes(i,1), bboxes(i,2), bboxes(i,3)-bboxes(i,1), bboxes(i,4)-bboxes(i,2)],...
                            'EdgeColor', color, 'LineWidth', 2, 'Parent', handles.imageViewAxes);
                        text(bboxes(i,1), bboxes(i,2)-5, num2str(scores(i), '%0.2f'), 'Color', color);
                    end
                else
                    warndlg(sprintf('No result for image %s is available.',imageName));
                end
            case 3 % seed detection
                imageName = selectedImageName;
                [~,imageBaseName,~] = fileparts(imageName);
                seedList = dir(fullfile(resultsDir,[imageBaseName '*']));
                if ~isempty(seedList)
                    name=fullfile(resultsDir,seedList(1).name);
                    [~,~,ext]=fileparts(name);
                    if strcmp(ext,'csv')||strcmp(ext,'txt')
                        seedData=importdata(name);
                        if ~isnumeric(seedData)
                            % text file contained header in seedData.textdata
                            seedData=seedData.data;
                        end
                        xs=round(seedData(:,1));
                        ys=round(seedData(:,2));
                    else
                        seeds = imread(fullfile(resultsDir,seedList(1).name));
                        p = regionprops(seeds, 'PixelIdxList');
                        [xs,ys] = ind2sub(size(seeds), cat(1,p.PixelIdxList));
                    end
                    axes(handles.imageViewAxes); hold on;
                    scatter(ys,xs,1,color,'Parent', handles.imageViewAxes);
                else
                    warndlg(sprintf('No result for image %s is available.',imageName));
                end
        case 4 % probability maps
            imageName = selectedImageName;
            [~,imageBaseName,~] = fileparts(imageName);
            probList = dir(fullfile(resultsDir,[imageBaseName '*.png']));
            if isempty(probList)
                probList = dir(fullfile(resultsDir,[imageBaseName '*.tiff']));
            end
            if ~isempty(probList)
                probMap = imread(fullfile(resultsDir,probList(1).name));
                probMap = im2double(probMap);
                if size(probMap,3)==1
                    colouredProbMap = cat(3, ones(size(probMap)).*(1-probMap), ones(size(probMap)).*probMap, zeros(size(probMap)));
                else
%                     colouredProbMap = probMap;
                    colouredProbMap = cat(3, ones(size(probMap(:,:,1))).*(1-probMap(:,:,1)), ones(size(probMap(:,:,1))).*probMap(:,:,1), zeros(size(probMap(:,:,1))));
                end
                axes(handles.imageViewAxes); hold on;
                h = imshow(colouredProbMap);
                set(h, 'AlphaData', ones(size(probMap,1),size(probMap,2))/4);
            else
                warndlg(sprintf('No result for image %s is available.',imageBaseName));
            end
    end
    uicontrol(handles.imageListBox);

    
function perim = bwlabelperim(labelledSegmentation)

er = imerode(labelledSegmentation ,strel('disk',1,4));
perim = er~=labelledSegmentation;

    
function perim = labelperim(labelledSegmentation)
perim = uint16(zeros(size(labelledSegmentation,1),size(labelledSegmentation,2)));
uv = unique(labelledSegmentation);
uv(uv==0) = [];

for l=reshape(uv,1,[])
    object = labelledSegmentation==l;
    objectPerim = bwperim(object);
    perim = perim + uint16(objectPerim)*l;
end
