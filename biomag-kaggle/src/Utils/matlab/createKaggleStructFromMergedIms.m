% creates kaggle-structure of images by copying all masks
% from 'like' and gray/colour images from 'new' to 'dest'

like='d:\Letöltés\SZBK munka cuccok\kaggle\progress\others_results\psfd\';
new='E:\train\conv3d\';
dest='d:\Letöltés\SZBK munka cuccok\kaggle\progress\others_results\psfd_0328\';
mkdir(dest);
l=dir(like);
l=l(3:end);

% for i=1:numel(l) mkdir(fullfile(dest,[l(i).name '.pngconv'],'images')); 
% mkdir(fullfile(dest,[l(i).name '.pngconv'],'masks')); copyfile(fullfile(like,l(i).name,'masks'),...
% fullfile(dest,[l(i).name '.pngconv'],'masks')); copyfile(fullfile(new,[l(i).name '.pngconv.png']),...
% fullfile(dest,[l(i).name '.pngconv'],'images',[l(i).name '.pngconv.png'])); end;

for i=1:numel(l)
    mkdir(fullfile(dest,sprintf('psfd_%03d',i),'images')); 
    mkdir(fullfile(dest,sprintf('psfd_%03d',i),'masks'));
    copyfile(fullfile(like,l(i).name,'masks'),...
        fullfile(dest,sprintf('psfd_%03d',i),'masks'));
    copyfile(fullfile(new,[l(i).name '.png']),...
        fullfile(dest,sprintf('psfd_%03d',i),'images',sprintf('psfd_%03d.png',i)));
end