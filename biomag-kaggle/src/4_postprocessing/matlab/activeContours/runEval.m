function scores=runEval(src,gt,writeFlag)
% Runs matlab eval script on images found in folder 'src' in the following
% struct: [src]\[imageName]\outStitched\[imageName.png]
% src: folder of images as above
% gt: folder of ground truth images as [gt]\*.png
% writeFlag: flag to indicate whether to write evaluation scores to a
% 'scores.csv' file under 'src'

l=dir(src);
l=l([l.isdir]);
l=l(3:end);
scores=zeros(numel(l),1);
if writeFlag
    f=fopen(fullfile(src,'scores.csv'),'w');
end
for i=1:numel(l)
    if contains(l(i).name,'_out_')
        continue;
    end
    name=[l(i).name '.png'];
    img=imread(fullfile(src,l(i).name,'outStitched',name));
    gtim=imread(fullfile(gt,name));
    scores(i,1)=evalImage(gtim,img);
    if writeFlag
        fprintf(f,'%s,%f\n',name,scores(i,1));
    end
end
if writeFlag
    fclose(f);
end
end