function postProcessAllData(inFolder, outFolder)
%% postprocess function

%% double size settings
conn = 8;
minSize = 40;
resizeFactor = 0.5;

fileList = dir([inFolder '*.tiff']);

mkdir([outFolder]);

for i=1:numel(fileList)
    
    inImg = imread([inFolder fileList(i).name]);
    
    outImg = postProcessKaggle(inImg, minSize, conn);
    
    outImg = imresize(outImg, resizeFactor, 'nearest');
    
    imwrite(outImg, [outFolder fileList(i).name]);
    
end
