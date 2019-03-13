classdef generateMasks < handle
    %Generate masks
    %   A class to encapsulate all the functions needed to generate the
    %   mask for a folder of initial segmentations
    
    properties
        DataBase
        inDir %the initial segmentation directory
        outDir %the directory for the generated masks
        extention % the extention of the files in the indir
        range       
        csvFile
    end
    
    methods
        function obj = generateMasks(DB,indir,outdir,extention)
            obj.DataBase = DB;
            obj.inDir = indir;
            obj.outDir = outdir;
            obj.extention = extention;
        end
        
        function generateMasksToFolder(obj,interval,imgSize)
            %INPUTS:
            %   interval    is an array of the indices to add to
            %               the artificial masks (i.e. interval = 1:20)     
            %               We generate length(interval) image to all input
            %               image in the folder
            %       WARNING: if the average cell diameter is over 40 pixel
            %       then we generate double amount of images to that
            %       cluster
            %   imgSize     The intended generated image size
            rng(19950927);            
            
            DB = obj.DataBase;            
            obj.range = interval;
            N = 750;
            subsamplingRatio = 0.7; %how much samples to throw away for subsampling the masks to avoid repetition
            
            measuredFeatures = {'ShapeRatio','Circularity','Eccentricity','Solidity'};
            ext = obj.extention;
            outdir = obj.outDir;
            pix2pixOutDir = fullfile(outdir, 'test')';
            grayScaleOutDir = fullfile(outdir,'masks');
            indir = obj.inDir;
            disp(['Out directory: ' outdir]);
            if ~exist(pix2pixOutDir,'dir'), mkdir(pix2pixOutDir); end
            if ~exist(grayScaleOutDir,'dir'), mkdir(grayScaleOutDir); end
            
            d = dir(fullfile(indir,['*.' ext]));
                                    
            for i=1:numel(d)
                
                [~,exExt,~] = fileparts(d(i).name);
                
                fprintf('(%d/%d) Generating artificial masks to: %s\n',i,numel(d),exExt);
                fprintf('Fetch similar vectors from database...\n');                
                
                testImg = imread(fullfile(indir,d(i).name));
                [estimatedAreas,vectors] = generateMasks.estimateVectorFromImage(testImg,measuredFeatures,DB);
                maskArray =  generateMasks.fetchMasks(measuredFeatures,vectors,estimatedAreas,N,DB);                                                                    
                
                % --------------------------------------
                %calc closest edges
                [edgeDistance,nofObj1] = generateMasks.estimateEdgeDistance(testImg);         
                % --------------------------------------
                r = sqrt(mean(estimatedAreas)/pi);
                if r>40
                    currentInterval = [interval interval(end)+1:(interval(end)+length(interval))]; %double the interval size
                else
                    currentInterval = interval;
                end
                generateMasks.writeOutMasks(currentInterval,exExt,maskArray,subsamplingRatio,edgeDistance,nofObj1,imgSize,pix2pixOutDir,grayScaleOutDir);
                
            end            
        end
        
        
        function [tensorSize,imageNames] = generateMasksToFolderSub(obj,interval,imgSize,subSamplingNumber)
            %INPUTS:
            %   interval            is an array of the indices to add to
            %                       the artificial masks (i.e. interval =
            %                       1:20)
            %       WARNING: if the average cell diameter is over 40 pixel
            %       then we generate double amount of images to that
            %       cluster
            %   imgSize             The intended image size to generate
            %   subSamplingNumber   is the number of images in the folder
            %                       that we use for artificial mask
            %        generation. Then alltogether we generate
            %        length(interval) number of images to the folder.
            %OUTPUTS:
            %   
            
            rng(19950927);               
            
            DB = obj.DataBase;            
            obj.range = interval;
            N = 750;
            subsamplingRatio = 0.7; %how much samples to throw away for subsampling the masks to avoid repetition
            
            measuredFeatures = {'ShapeRatio','Circularity','Eccentricity','Solidity'};
            ext = obj.extention;
            outdir = obj.outDir;
            pix2pixOutDir = fullfile(outdir,'test');
            grayScaleOutDir = fullfile(outdir,'grayscale');
            indir = obj.inDir;
            if ~exist(pix2pixOutDir,'dir'), mkdir(pix2pixOutDir); end
            if ~exist(grayScaleOutDir,'dir'), mkdir(grayScaleOutDir); end
            
            d = dir(fullfile(indir,['*.' ext]));
                        
            if isinf(subSamplingNumber) || subSamplingNumber > numel(d)
                realizedSSN = numel(d);            
            else
                realizedSSN = subSamplingNumber;                
            end
            order = randperm(numel(d));
            jointAreas = cell(realizedSSN,1);
            jointVectors = cell(realizedSSN,1);
            jointEdgeD = cell(realizedSSN,1);
            jointNum = cell(realizedSSN,1);
            
            imageNames = cell(realizedSSN,1);            
            for ii=1:realizedSSN
                i = order(ii);
                
                [~,exExt,~] = fileparts(d(i).name);
                
                fprintf('(%d/%d) Measuring properties of: %s\n',ii,realizedSSN,exExt);
                testImg = imread(fullfile(indir,d(i).name));
                [estimatedAreas,vectors] = generateMasks.estimateVectorFromImage(testImg,measuredFeatures,DB);
                jointAreas{ii} = estimatedAreas;
                jointVectors{ii} = vectors;    
                [edgeDistance, ~] = generateMasks.estimateEdgeDistance(testImg);
                jointEdgeD{ii} = edgeDistance;
                if ~isempty(estimatedAreas)
                    jointNum{ii} = prod(imgSize)/mean(estimatedAreas);
                else                    
                    jointNum{ii} = 0;
                end
            end         
                            
            %clear out empty images from vectors
            delIdx = logical(cellfun(@isempty,jointVectors));
            jointVectors(delIdx) = [];
            jointAreas(delIdx) = [];
            jointEdgeD(delIdx) = [];
            jointNum(delIdx) = [];

            
            fprintf('Fetch similar vectors from database...\n');
            if ~isempty(cell2mat(jointVectors)) && ~isempty(cell2mat(jointVectors))
                maskArray =  generateMasks.fetchMasks(measuredFeatures,cell2mat(jointVectors),cell2mat(jointAreas),N,DB);                           
            else
                maskArray = cell(1,2*N);
                for i=1:2*N
                    maskArray{i} = zeros(2);
                end
            end

            r = sqrt(mean(cell2mat(jointAreas))/pi);
            if r>40
                interval = [interval interval(end)+1:(interval(end)+length(interval))]; %double the interval size
            end
            splitToFolders = strsplit(outdir,filesep);
            generateMasks.writeOutMasks(interval,splitToFolders{end},maskArray,subsamplingRatio,cell2mat(jointEdgeD),cell2mat(jointNum),imgSize,pix2pixOutDir,grayScaleOutDir);            

        end                                
        
        function [tensorSize,imageNames] = generateMasksToFolderSubCyto(obj,interval,imgSize,subSamplingNumber)
            %INPUTS:
            %   interval            is an array of the indices to add to
            %                       the artificial masks (i.e. interval =
            %                       1:20)
            %       WARNING: if the average cell diameter is over 40 pixel
            %       then we generate double amount of images to that
            %       cluster
            %   imgSize             The intended image size to generate
            %   subSamplingNumber   is the number of images in the folder
            %                       that we use for artificial mask
            %        generation. Then alltogether we generate
            %        length(interval) number of images to the folder.
            %OUTPUTS:
            %   
            
            rng(19950927);               
            
            DB = obj.DataBase;            
            obj.range = interval;
            N = 750;
            subsamplingRatio = 0.7; %how much samples to throw away for subsampling the masks to avoid repetition
            
            measuredFeatures = {'ShapeRatio','Circularity','Eccentricity','Solidity'};
            ext = obj.extention;
            outdir = obj.outDir;
            pix2pixOutDir = fullfile(outdir,'test');
            grayScaleOutDir = fullfile(outdir,'grayscale');
            indir = obj.inDir;
            if ~exist(pix2pixOutDir,'dir'), mkdir(pix2pixOutDir); end
            if ~exist(grayScaleOutDir,'dir'), mkdir(grayScaleOutDir); end
            
            d = dir(fullfile(indir,['*.' ext]));
                        
            if isinf(subSamplingNumber) || subSamplingNumber > numel(d)
                realizedSSN = numel(d);            
            else
                realizedSSN = subSamplingNumber;                
            end
            order = randperm(numel(d));
            jointAreas = cell(realizedSSN,1);
            jointVectors = cell(realizedSSN,1);
            jointEdgeD = cell(realizedSSN,1);
            jointNum = cell(realizedSSN,1);
            
            imageNames = cell(realizedSSN,1);            
            for ii=1:realizedSSN
                i = order(ii);
                
                [~,exExt,~] = fileparts(d(i).name);
                
                fprintf('(%d/%d) Measuring properties of: %s\n',ii,realizedSSN,exExt);
                testImg = imread(fullfile(indir,d(i).name));
                [estimatedAreas,vectors] = generateMasks.estimateVectorFromImage(testImg,measuredFeatures,DB);
                jointAreas{ii} = estimatedAreas;
                jointVectors{ii} = vectors;    
                [edgeDistance, ~] = generateMasks.estimateEdgeDistance(testImg);
                jointEdgeD{ii} = edgeDistance;
                if ~isempty(estimatedAreas)
                    jointNum{ii} = prod(imgSize)/mean(estimatedAreas);
                else                    
                    jointNum{ii} = 0;
                end
            end         
                            
            %clear out empty images from vectors
            delIdx = logical(cellfun(@isempty,jointVectors));
            jointVectors(delIdx) = [];
            jointAreas(delIdx) = [];
            jointEdgeD(delIdx) = [];
            jointNum(delIdx) = [];

            
            fprintf('Fetch similar vectors from database...\n');
            if ~isempty(cell2mat(jointVectors)) && ~isempty(cell2mat(jointVectors))
                maskArray =  generateMasks.fetchMasksCyto(measuredFeatures,cell2mat(jointVectors),cell2mat(jointAreas),N,DB);                           
            else
                maskArray = cell(1,2*N);
                for i=1:2*N
                    maskArray{i} = zeros(2);
                end
            end

            r = sqrt(mean(cell2mat(jointAreas))/pi);
            if r>40
                interval = [interval interval(end)+1:(interval(end)+length(interval))]; %double the interval size
            end
            splitToFolders = strsplit(outdir,filesep);
            generateMasks.writeOutMasks(interval,splitToFolders{end},maskArray,subsamplingRatio,cell2mat(jointEdgeD),cell2mat(jointNum),imgSize,pix2pixOutDir,grayScaleOutDir);            

        end
        
        function sortToFolders(obj,targetDir,csvFile)
            inputDir = obj.outDir;
            obj.csvFile = csvFile;
            %targetDir = '/home/biomag/szkabel/180330_MORE';
            %csvFile = '/home/biomag/szkabel/styles.csv';

            sortToFolders(inputDir,targetDir,csvFile);
        end
        
        function copyBackImagesFromStyleTransfer(obj, styledDir)
            %Copies back the result of the style transfer to the kaggle formatted
            %folders created beforehand by generateMasksToFolder
            
            csvFile = obj.csvFile;
            %styledDir = '/home/biomag/tivadar/pytorch-CycleGAN-and-pix2pix/results';
            
            ext = obj.extention;
            indir = obj.inDir;            
            outdir = obj.outDir;
            interval = obj.range;            
            
            d = dir(fullfile(indir,['*.' ext]));
            
            T = readtable(csvFile); %#ok<*PROPLC>
            
            for i=1:numel(d)            
                [~,exExt,~] = fileparts(d(i).name);
                for j=interval
                    generatedName = [exExt '_' num2str(j,'%02d') ];
                    curImgDir = fullfile(outdir,generatedName);
                    for k=1:length(T.Style)
                        if strcmp([exExt '.png'],T.Name{k})
                            target = T.Style(k);
                            tcd = fullfile(curImgDir,'images');
                            if ~isdir(tcd), mkdir(tcd); end
                            try
                                copyfile(...
                                    fullfile(styledDir,num2str(target),'test_latest','images',[generatedName '_fake_B.png']),...
                                    fullfile(tcd,[generatedName '.png']));
                            catch e
                                disp(getReport(e));
                                disp(fullfile(styledDir,num2str(target),'test_latest','images',[generatedName '_fake_B.png']));
                                rmdir(curImgDir,'s');
                            end

                            break;
                        end
                    end
                end
            end
            
        end
                
    end
    
    methods (Static)
        function [edgeDistance,nofObj1] = estimateEdgeDistance(testImg)
            %estimates the average closest distance between masks on testImg
            testImg = relabelImage(testImg);
            mR = regionprops(testImg,'Centroid');
            m1.Centers = cat(1,mR.Centroid);
            nofObj1 = size(m1.Centers,1);
            m1.Mask = double(testImg);
            [m1.MaxRadius,m1.PixelChains] = extractPixelChainsFromMask(m1.Mask,m1.Centers,nofObj1);
            
            relationFeatures = zeros(nofObj1,6);
            for j=1:nofObj1
                m2 = m1;
                %in mask1 retain only the object i and set its value to
                %1.
                mask1 = m1.Mask;
                maskedMask1 = zeros(size(mask1));
                maskedMask1(mask1 == j) = 1;
                %delete object i from mask2
                m2.Mask(mask1 == j) = 0;
                %shift the indices one down to fill the gap caused by
                %deletion of object i.
                m2.Mask(mask1 > j) = mask1(mask1 > j) - 1;
                %update other fields of m2 as well
                m2.Centers     =     m2.Centers([1:j-1 j+1:end],:);
                m2.MaxRadius   =   m2.MaxRadius([1:j-1 j+1:end]);
                m2.PixelChains = m2.PixelChains([1:j-1 j+1:end]);
                %create mm1 the current new mask for the single object
                mm1.Mask = maskedMask1;
                mm1.Centers = m1.Centers(j,:);
                mm1.MaxRadius = m1.MaxRadius(j);
                mm1.PixelChains = m1.PixelChains(j);
                
                [relationFeatures(j,:),~] = measureObjectRelation(mm1,m2,1);
            end
            
            edgeDistance = relationFeatures(:,4);              
            
        end
        
        
        function maskArray =  fetchMasks(measuredFeatures,vectors,estimatedAreas,N,DB)
            %fetch both from DB and generate the artificial masks
            [fID,iID,rID,~] = DB.fetchCloseIndex(measuredFeatures,vectors,N);
            
            fetchedMasks = DB.fetchMasks(fID,iID,rID);
            
            fetchedMasks = clearHoles(fetchedMasks);
            
            fetchedMasks = resizeMasks(fetchedMasks,estimatedAreas);
            
            nuclei = generate_n_object(2*N-length(fetchedMasks),[estimatedAreas vectors],[0.1 0.1],[2 2 5 0.2]);                                
            maskArray = cell(1,2*N);
            maskArray(1:2:2*length(fetchedMasks)-1) = fetchedMasks;

            %Fill in gaps with generated masks
            k = 1;
            for j=1:2*N
                if isempty(maskArray{j}) && k<=length(nuclei)
                    maskArray{j} = double(getshape(nuclei(k)));
                    k = k+1;
                end
            end       
        end
        
        % added cytoplasm generation function
        function maskArray =  fetchMasksCyto(measuredFeatures,vectors,estimatedAreas,N,DB)
            %fetch both from DB and generate the artificial masks
            [fID,iID,rID,~] = DB.fetchCloseIndex(measuredFeatures,vectors,N);
            
            fetchedMasks = DB.fetchMasks(fID,iID,rID);
            
            fetchedMasks = clearHoles(fetchedMasks);
            
            fetchedMasks = resizeMasks(fetchedMasks,estimatedAreas);
            
%             nuclei = generate_n_object_cyto(2*N-length(fetchedMasks),[estimatedAreas vectors],[0.3 0.06],[2 2 5 0.2]);                                
            nuclei = generate_n_object_cyto(2*N-length(fetchedMasks),[estimatedAreas vectors],[0.3 0.7],[2 2 5 0.2]);
            
            maskArray = cell(1,2*N);
            maskArray(1:2:2*length(fetchedMasks)-1) = fetchedMasks;

            %Fill in gaps with generated masks
            k = 1;
            for j=1:2*N
                if isempty(maskArray{j}) && k<=length(nuclei)
                    maskArray{j} = double(getshape(nuclei(k)));
                    k = k+1;
                end
            end       
        end
        
        function [estimatedAreas,vectors] = estimateVectorFromImage(testImg,measuredFeatures,DB)
            dropOutSize = 35;
            vectors = DB.measureCellProperties(testImg,measuredFeatures);
            estimatedAreas = DB.measureCellProperties(testImg,{'Area'});            
            filterArray = estimatedAreas<dropOutSize;
            estimatedAreas(filterArray) = [];
            vectors(filterArray,:) = [];
        end
        
        function writeOutMasks(interval,nameBase,maskArray,subsamplingRatio,edgeDistances,numbers,imgSize,p2pDir,grayDir)
            if ~exist(p2pDir,'dir'), mkdir(p2pDir); end
            fprintf('\nSave images\n')
            minNumberOfCells = 25;
            touchRatio = 0.05;
            if numel(numbers) == 1
                meanN = numbers;
                stdN = 10;
            else
                meanN = mean(numbers);
                stdN = std(numbers);
            end
            for j=interval
                generatedName = [nameBase '_' num2str(j,'%03d') ];                
                
                subsampling = rand(1,length(maskArray));
                tmpMasks = maskArray(subsampling>subsamplingRatio);
                genImg = tetrisImage(...
                    tmpMasks,...
                    touchRatio,...
                    imgSize,...
                    round(mean(edgeDistances)+2*std(edgeDistances)),...
                    max( minNumberOfCells,round(normrnd(meanN,stdN)) )    );
                imwrite(uint16(genImg), fullfile(grayDir, [generatedName '.tiff']));
                transformedToStyleFormat = repmat(uint8(genImg>0)*255,1,2,3);
                imwrite(transformedToStyleFormat,fullfile(p2pDir,[generatedName '.png']));
                fprintf('#');
                
            end
            fprintf('\n');
        end
    end
    
end

