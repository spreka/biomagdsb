classdef FeatureDataBase < handle
    %FeatureDataBase:
    %   stores features of many really annotated nuclei.
    %   stored as reference
    
    properties
        folders
        %for storing the folders where data comes from
        
        propsInDB
        %a list of features that are stored in the DB
        
        features
        %cellarray exactly same long as "folders". Each entry is another
        %cellarray that contains the images for the given folder (matching
        %the same index in folders) Then a single entry for that cellarray
        %stores the cell properties coming from
        
        BoundingBoxes
        %cellarray similar to features in structure. It is as long as
        %folders and then inside each entry is an image. In there the
        %result is the structure with regionprops for BoundingBoxes
    end
    
    methods
        function obj = FeatureDataBase()
        %Constructor, just for init
        
                obj.folders = {};
                obj.propsInDB = FeatureDataBase.measureCellProperties();
                obj.features = {};
                obj.BoundingBoxes = {};
        end
        
        function addFolderToDataBase(obj,folder,imgExt)
            if nargin<3
                imgExt = 'png';
            end
            
            d = dir([folder filesep '*.' imgExt]);
            
            folderID = length(obj.folders)+1;
            obj.folders{folderID}.path = folder;
            obj.folders{folderID}.imgs = {d.name};
            obj.features{folderID} = {};
            obj.BoundingBoxes{folderID} = {};
            
            nofHashes = 17; currHash = 0; %cmd waitbar
            fprintf('Measuring images:\n');
            for i=1:numel(d)
                %This imread block is also in fetch single mask member
                %function. If this is modified that one also has to be.
                % ******************************** %
                img = imread(fullfile(folder,d(i).name));
                img = clearBorderObjects(img);
                img = bwlabel(img);
                % ******************************** %
                res = FeatureDataBase.measureCellProperties(img);
                obj.features{folderID}{i} = res;
                obj.BoundingBoxes{folderID}{i} = regionprops(img,'BoundingBox');
                if (i/numel(d)*nofHashes>currHash), currHash = currHash+1; fprintf('#');  end %cmd waitbar
            end
            fprintf('\n');
            
        end
        
        function [folderIdx,imgIdx,rowIdx,hitVectors] = fetchCloseIndex(obj,props,vectors,N)
            %Searches for the closest shapes within the database. Props is
            %a cellarray of strings (listed by measureCellProperties)
            %describing the meaning the columns of the vectors. The rows
            %of vectors then are the samples to which we search the closest
            %objects
            %OUTPUT is an array of indices with length N identifying the
            %index of the folder the image and the object number for the
            %closest N object.
            
            featureIdx = obj.getFeatureIdx(props);
            
            allImgID = [];
            allRowID = [];
            allHitVectors = [];
            folderIdx = [];
            for i=1:length(obj.folders)
                [imgID,rowID,hitVectors] = obj.fetchFromFolder(i,vectors,N,featureIdx);
                allImgID = [allImgID; imgID];
                allRowID = [allRowID; rowID];
                allHitVectors = [allHitVectors; hitVectors];
                folderIdx(end+1:length(allRowID)) = i;
            end
            
            closestIdx = FeatureDataBase.getClosesestIndices(allHitVectors,vectors,N);
            
            folderIdx = folderIdx(closestIdx);
            rowIdx = allRowID(closestIdx);
            imgIdx = allImgID(closestIdx);
            hitVectors = allHitVectors(closestIdx,:);
            
            
        end                
        
        function [imgID,rowID,hitVectors] = fetchFromFolder(obj,folderIdx,vectors,N,featureIdx)
            featuresInFolder = cell2mat(obj.features{folderIdx}');
            featuresInFolder = featuresInFolder(:,featureIdx);
            imgIDs = zeros(size(featuresInFolder,1),1);
            rowIDs = zeros(length(imgIDs),1);
            
            %fill out imgIDs and rowIDs
            iID = 1;
            j = 1;
            sums = cellfun('size',obj.features{folderIdx},1);
            for i=1:length(rowIDs)
                imgIDs(i) = iID;
                rowIDs(i) = j;
                j = j+1;
                while iID<length(sums) && sum(sums(1:iID))==i
                    iID = iID+1;
                    j = 1;
                end
            end
            
            closestIdx = FeatureDataBase.getClosesestIndices(featuresInFolder,vectors,N);
            
            imgID = imgIDs(closestIdx);
            rowID = rowIDs(closestIdx);
            hitVectors = featuresInFolder(closestIdx,:);
            
        end
        
        function featureIdx = getFeatureIdx(obj,props)
             featureIdx = FeatureDataBase.getStaticFeatureIdx(props,obj.propsInDB);
        end
        
        function maskArray = fetchMasks(obj,folderIdx,imgIdx,rowIdx)
            %folderIdx, imgIdx, rowIdx must be equally long array
            n = numel(folderIdx);
            maskArray = cell(1,n);
            for i=1:n
                maskArray{i} = obj.fetchSingleMask(folderIdx(i),imgIdx(i),rowIdx(i));
                if isempty(maskArray{i})
                    disp('stop');
                end
            end
        end
        
        function singleMask = fetchSingleMask(obj,fID,iID,rID)
            %This imread block is also in add folder to database member
            %function. If this is modified that one also has to be.
            % ******************************** %
            img = imread(fullfile(obj.folders{fID}.path,obj.folders{fID}.imgs{iID}));            
            img = clearBorderObjects(img);
            img = bwlabel(img);
            % ******************************** %
            R = obj.BoundingBoxes{fID}{iID};
            rowS = round(R(rID).BoundingBox(2));
            rowL = round(R(rID).BoundingBox(4));
            colS = round(R(rID).BoundingBox(1));
            colL = round(R(rID).BoundingBox(3));
            singleMask = img(rowS:rowS+rowL-1,colS:colS+colL-1);
            %clearing out all unnecessary part in the mask
            singleMask(singleMask~=rID) = 0;
            %make it binary
            singleMask = double(singleMask>0);
        end
    end
    
    methods (Static)
        function [res] = measureCellProperties(img,props)
            %measures the properties of the img. Gives back a matrix, where
            %the rows corresponds to observations columns to measurements
            %(properties, features) If img is not provided then res gives
            %back the column headers (what type of features are provided)
            %
            %   Area: the area of the cells
            %   ShapeRatio: Major/Minor axis ratio
            %   Circularity
            
            regProps = {'Area','MajorAxisLength','MinorAxisLength','Eccentricity','Solidity','Perimeter'};
            names = [regProps {'Circularity','ShapeRatio'}];
            if nargin<1                                
                res = names;
                return;
            end                     
            
            if nargin<2
                props = names;                            
            end
            idx = FeatureDataBase.getStaticFeatureIdx(props,names);
            
            R = regionprops(img,regProps);
            
            res = zeros(numel(R),numel(names));
            
            for i=1:numel(regProps)
                res(:,i) = [R.(regProps{i})];
            end
            %Circularity
            i = numel(regProps)+1;
            res(:,i) = [R.Area] ./ ([R.Perimeter]).^2;
            %ShapeRatio
            i = i+1;
            res(:,i) = [R.MajorAxisLength] ./ [R.MinorAxisLength];
            
            res = res(:,idx);
        end
        
       function [idx] = getClosesestIndices(database,target,N)
            %gives back which are the N closest index to the targets
            distM = pdist2(database,target);
            closestDist = min(distM,[],2);
            % #TODO
            % OPTIONALLY SET DISTANCE LIMIT on closestDistance
            idx = FeatureDataBase.getTopHits(closestDist,N);
       end 
        
       function [idx,value] = getTopHits(v,N)
           if compareVersions('(R2017b)',version)
               [value,idx] = mink(v,N);
           else
               [value,idx] = sort(v,'ascend');
               if N<length(v)
                   idx = idx(1:N);
                   value = value(1:N);
               end
           end
       end
       
       function featureIdx = getStaticFeatureIdx(props,names)
           %get the indices of props within names
           featureIdx = zeros(1,length(props));            
            for i=1:length(props)
                for j=1:length(names)
                    if strcmp(names{j},props{i})
                        featureIdx(i) = j;
                    end
                end
            end           
       end
       
    end
    
end

