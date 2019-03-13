function cellSizeDataGenerate(inFolder,outFolder)

%disp(inFolder)
%disp(outFolder)
MAX_TENSOR = 6000;
TENSOR_SORT = 1;
TENSOR_STEP_SIZE = 64;

mkdir(outFolder);

for correctionFactor =[1 2 4]
    %% estimate cell scaling factor and write to a csv
    outFileRCNN = ['scales-rcnn-' num2str(correctionFactor) 'x.csv'];
    outFileUNET = 'scales-unet.csv';
    expectedCellSize = 40;
    
    
    
    fileList = dir([inFolder '*.tiff']);
    
    
    outFUNET = fopen([outFolder filesep outFileUNET], 'w');
    outFrcnn = fopen([outFolder filesep outFileRCNN], 'w');
    
    for i=1:numel(fileList)
        
        maskImg = imread([inFolder fileList(i).name]);
        [sx, sy] = size(maskImg);
        [median_size, std_size] = estimateCellSizeFromMask(maskImg);
        resizeTo = min((expectedCellSize/median_size)*((sx+sy)/2)*correctionFactor, MAX_TENSOR);
        if ~TENSOR_SORT
            fprintf(outFrcnn, '%s\t%f\n', fileList(i).name(1:end-5), (expectedCellSize/median_size)*((sx+sy)/2)*correctionFactor);
        end
        resizeToList(i) = resizeTo ;
        
        fprintf(outFUNET, '%s\t%f\n', fileList(i).name(1:end-5), (expectedCellSize/median_size)*correctionFactor);
        
    end
    
    if TENSOR_SORT
        [sv, si] = sort(resizeToList);
        % bring to the closest TENSOR_STEP_SIZE
        sv = round(sv/(TENSOR_STEP_SIZE*correctionFactor)) * (TENSOR_STEP_SIZE*correctionFactor);
        for i=1:numel(fileList)
            fprintf(outFrcnn, '%s\t%f\n', fileList(si(i)).name(1:end-5), sv(i));
        end
    end
    
    fclose(outFUNET);
    fclose(outFrcnn);
end