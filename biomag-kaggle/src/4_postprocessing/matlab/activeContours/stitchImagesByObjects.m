function stitchImagesByObjects(src,pad)
% Runs stitchImageByObjects.m for each image found under folder 'src'.
% src: folder of small images to stitch together as
% [src]\[name]\out\[name*.png]
% pad: padding size in pixels
% 
% Example:
%     src='d:\Letöltés\SZBK munka cuccok\kaggle\progress\2dseg\testing\';
%     pad=40;
%     stitchImagesByObjects(src);

l=dir(src);
l=l([l.isdir]);
l=l(3:end);
tmp=strsplit(src,filesep);
collectedOut=fullfile(src,'_out_',tmp{end-1});
for i=1:numel(l)
    failed=stitchImageByObjects(src,l(i).name,pad);
    if failed
        fprintf('failed %s\n',l(i).name);
    else
        if ~exist(collectedOut,'dir')
            mkdir(collectedOut);
        end
        copyfile(fullfile(src,l(i).name,'outStitched\',[l(i).name '.png']),...
            fullfile(collectedOut,[l(i).name '.png']));
    end
end
end