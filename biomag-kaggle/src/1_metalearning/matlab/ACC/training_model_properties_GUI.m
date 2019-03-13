function varargout = training_model_properties_GUI(varargin)
% TRAINING_MODEL_PROPERTIES_GUI MATLAB code for training_model_properties_GUI.fig
%      TRAINING_MODEL_PROPERTIES_GUI, by itself, creates a new TRAINING_MODEL_PROPERTIES_GUI or raises the existing
%      singleton*.
%
%      H = TRAINING_MODEL_PROPERTIES_GUI returns the handle to a new TRAINING_MODEL_PROPERTIES_GUI or the handle to
%      the existing singleton*.
%
%      TRAINING_MODEL_PROPERTIES_GUI('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in TRAINING_MODEL_PROPERTIES_GUI.M with the given input arguments.
%
%      TRAINING_MODEL_PROPERTIES_GUI('Property','Value',...) creates a new TRAINING_MODEL_PROPERTIES_GUI or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before training_model_properties_GUI_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to training_model_properties_GUI_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help training_model_properties_GUI

% Last Modified by GUIDE v2.5 02-Feb-2017 14:38:26

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @training_model_properties_GUI_OpeningFcn, ...
                   'gui_OutputFcn',  @training_model_properties_GUI_OutputFcn, ...
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


% --- Executes just before training_model_properties_GUI is made visible.
function training_model_properties_GUI_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to training_model_properties_GUI (see VARARGIN)

% Choose default command line output for training_model_properties_GUI
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes training_model_properties_GUI wait for user response (see UIRESUME)
% uiwait(handles.figure1);

calculateAccuracy(handles,varargin{1});

% --- Outputs from this function are returned to the command line.
function varargout = training_model_properties_GUI_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;

function calculateAccuracy(handles,CommonHandles)
    
    
    % get the predicted labels from sac with CrossValid
    [predicted,~] = sacCrossValidate(CommonHandles.SALT.model,CommonHandles.SALT.trainingData,10,1);
    real = CommonHandles.SALT.trainingData.labels';
    
    eq = 0;
    neq = 0;
    
    % count the difference of true and false predictions
    for i=1:length(real)
        if predicted(i) == real(i)
            eq = eq + 1;
        else
            neq = neq + 1;
        end
    end
    
    set(handles.text10,'String',eq);
    set(handles.text12,'String',neq);
    set(handles.text19,'String',(neq+eq));
    set(handles.text21,'String',CommonHandles.ClassifierNames{CommonHandles.SelectedClassifier});
    set(handles.text23,'String',size(CommonHandles.SALT.trainingData.instances,2));
    %set(handles.text25,'String',sprintf('%0.2f sec',CommonHandles.TrainingTime));
    
    % calculate the accuracy and show it on figure
    correct_accuracy = eq / length(real) * 100;
    set(handles.text3,'String',sprintf('%0.3f %%',correct_accuracy));
    incorrect_accuracy = 100 - correct_accuracy;
    set(handles.text17,'String',sprintf('%0.3f %%',incorrect_accuracy));
    
    
    % calculate the confusions
    [C,order] = confusionmat(real,predicted);
    
    % clear the 0th row if exists
    C_length = length(C);
    if order(1) == 0
        C = C(2:C_length,2:C_length);
    end
    
    % set the confusion columns to 40
    columnW = cell(1,size(C,1));
    for i=1:length(columnW) columnW{i}=40; end
    
    t = uitable('Parent', handles.figure1, 'Position', [20 10 650 220],'Data', C,'ColumnWidth','auto','ColumnWidth',columnW);
    
    % calculate precision and recall
    data = [];
    for i=1:size(C,1)
        tp = C(i,i);
        fp = sum(C(:,i))-C(i,i);
        fn = sum(C(i,:),2)-C(i,i);
        data(i,1) = tp / (tp + fp); % precision
        data(i,2) = tp / (tp + fn); % recall / TPR
        data(i,3) = 2 * tp / (2*tp + fp + fn); % F1 score
    end
    
    b = uitable('Parent', handles.figure1, 'Position', [380 300 250 200],'ColumnName',{'Recall'; 'Precision'; 'F1-score'},'ColumnWidth',{60 60 60},'Data',data);
    
