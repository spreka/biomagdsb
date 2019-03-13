function checkInitialSegmentation(inputDir,initialSegmentation,inExt,segExt)
% checks if an initial segmentation of the input image exists in the
% initial segmentation folder; if not it moves the image from its cluster

mkdir(fullfile(inputDir,'missingSegmentation'));

l=dir(fullfile(inputDir,'group_*'));
for i=1:numel(l)
    cur=fullfile(inputDir,l(i).name);
    segs=dir(fullfile(initialSegmentation,['*.' segExt]));
    segNames={segs(:).name};
    segNames=cellfun(@(x) x(1:end-numel(segExt)-1),segNames,'UniformOutput',false);
    files=dir(fullfile(cur,['*.' inExt]));
%     fileNames=cellfun(@(x) x(1:end-numel(segExt)-1),{files(:).name},'UniformOutput',false);
    for j=1:numel(files)
        if isempty(find(~cellfun(@isempty,strfind(segNames,files(j).name(1:end-numel(inExt)-1))),1))
            if ~exist(fullfile(inputDir,'missingSegmentation',l(i).name),'dir')
                mkdir(fullfile(inputDir,'missingSegmentation',l(i).name));
            end
            movefile(fullfile(cur,files(j).name),fullfile(inputDir,'missingSegmentation',l(i).name,files(j).name));
        end
    end
end

end