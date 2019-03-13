function postProcessAllDataFINAL(inFolder, outFolder, minSize)
%% postprocess function

%% double size settings
conn = 8;
resizeFactor = 0.5;

fileList = dir([inFolder '*.tiff']);

mkdir([outFolder]);

for i=1:numel(fileList)
    
    inImg = imread([inFolder fileList(i).name]);
    
    outImg = postProcessKaggle(inImg, minSize, conn);
    
    outImg = imresize(outImg, resizeFactor, 'nearest');
    
    imwrite(outImg, [outFolder fileList(i).name]);
    
end
