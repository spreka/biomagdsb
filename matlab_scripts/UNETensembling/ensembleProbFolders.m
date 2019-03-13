function ensembleProbFolders(inFolder, outFolder)
% inFolder contains different UNet predictions

d = dir(inFolder);

inFolders = {};
for i=1:length(d)
    if ~(strcmp(d(i).name,'.') || strcmp(d(i).name,'..'))
        inFolders{end+1} = fullfile(inFolder, d(i).name);
    end
end
if ~exist(outFolder,'dir')
    mkdir(outFolder);
end
fileList = dir(fullfile(inFolders{1},'images', '*.png'));
for i=1:numel(fileList)
    for j=1:numel(inFolders)
       
        if j == 1
            out = double(imread(fullfile(inFolders{j},'images', fileList(i).name)));
        else
            out = out + double(imread(fullfile(inFolders{j},'images', fileList(i).name)));
        end        
    end
    out = out / numel(inFolders);
    imwrite(uint16(out), [outFolder fileList(i).name]);
end