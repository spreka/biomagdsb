function [features,featureNames,imageNames] = loadBasicFeatures(origDir,recalc,saveFile)
%[features,featureNames,imageNames] = loadBasicFeatures(origDir,recalc,saveFile)
%   Loads in basic CellProfiler features specified in featureTypes for all
%   images in origDir folder.
%   It extracts RGB channels and also transforms to LAB space, and measuers
%   all features on all the images
%       INPUT:
%           origDir: folder for the original (training) images
%           recalc: bool to force recalculation of features
%           save:   bool to save data to disk in mat and in csv
%       OUTPUT:
%           features a cellarray that contains for each image its feature
%                     vector. One image corresponds to an entry and it is a
%                     row vector
%           featureNames 
%           imageNames a cellarray exactly as long as many rows the
%                       features output has containing the imageIDs.

d = dir([origDir filesep '*.png' ]);
matFileName = fullfile(origDir,'basicFeatures.mat');

%calc features only if they are not calculated already
if exist(matFileName,'file') && ~recalc
    load(matFileName);
else
    nofIm = length(d);
    imageNames = cell(1,nofIm);
    features = cell(nofIm,1);      
    
    i=1; % the first separately to get featureNames
    imageNames{i} = d(i).name;
    [currFeature,featureNames] = runPipe(d(i),origDir);
    features{i} = currFeature;
        
    parfor i=2:nofIm 
        imageNames{i} = d(i).name;
        [currFeature,~] = runPipe(d(i),origDir);
        features{i} = currFeature;
    end
           
    if saveFile
        save(matFileName,'features','featureNames','imageNames');
        f = fopen(fullfile(origDir,'featureNames.csv'),'w');
        for i=1:length(featureNames)
            fprintf(f,'%s\n',featureNames{i});
        end
        fclose(f);       
        csvwrite(fullfile(origDir,'features.csv'),cell2mat(features));
    end         
        
end

end

function [currFeat,featureNames] = runModuleOnChannels(featureType,imgToProc,chanIndices,chanNames,MatrixLabelImage,optSettings)
    handles.Settings.VariableValues = defaultSettingsForModule(featureType);   
    if nargin > 5
        for i=1:length(optSettings)
            handles.Settings.VariableValues{1,optSettings{i}.id} = optSettings{i}.value;
        end
    end
    featureNames = '';
    currFeat = [];
    for k = chanIndices
        [currF,currN] = feval(featureType,handles,imgToProc{k},MatrixLabelImage);       
        currFeat = [currFeat currF];        
        for l=1:length(currN)
            featureNames{(k-1)*length(currN)+l} = [chanNames{k} '_' currN{l}];
        end        
    end
end

function [currFeature,featureNames] = runPipe(di,origDir)    
        [currImg,map] = imread(fullfile(origDir,di.name));
        if size(currImg,3)~=3 && size(map,2)==3
            currImg = ind2rgb(currImg,map);
        else
            currImg = im2double(currImg);
        end
        disp(['Process image: ' di.name]);
        imgToProc = cell(1,6); %in any case we must have same number of channels
        if size(currImg,3)~=3            
            currImg = repmat(currImg,1,1,3);                                    
        end
        for j=1:3
            imgToProc{j} = currImg(:,:,j);
        end
        % {
        if size(currImg,3) == 3
            labImg = rgb2lab(currImg);
        end
        for j=1:3
            imgToProc{j+3} = labImg(:,:,j);
        end
        % }
        chanNames = {'red','green','blue','Lightness','a','b'};                        
        MatrixLabelImage = ones(size(currImg,1),size(currImg,2));
                                     
        featureNames = {'width','height'};
        currFeature = [size(currImg,2),size(currImg,1)];
       
        [currF, currN] = runModuleOnChannels('MeasureObjectIntensity',imgToProc,1:length(chanNames),chanNames,MatrixLabelImage);
        featureNames = [featureNames currN];
        currFeature = [currFeature currF];
        [currF, currN] = runModuleOnChannels('MeasureTexture',imgToProc,1:length(chanNames),chanNames,MatrixLabelImage);
        featureNames = [featureNames currN];
        currFeature = [currFeature currF];                        
        [currF, currN] = runModuleOnChannels('MeasureTexture',imgToProc,1:length(chanNames),chanNames,MatrixLabelImage,{struct('id',8,'value','5')});
        featureNames = [featureNames currN];
        currFeature = [currFeature currF];                
end

