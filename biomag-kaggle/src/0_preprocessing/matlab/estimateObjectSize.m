function [predSize,curMajax]=estimateObjectSize(prob,name,show)
% Estimates object size for an image based on Otsu thresholding of
% probability maps.
% prob: folder of probability maps
% name: name or id of image
% show: flag to indicate to show images and result size

if nargin<3
    show=false;
end

[~,n,~]=fileparts(name);
name=[n '.png'];
% Otsu's thresholding method
img=im2double(imread(fullfile(prob,name)));
bin=img>graythresh(img);
% get major axis length as estimate of object size
r=regionprops(bin,'MajorAxisLength');
majax=[r.MajorAxisLength]';
curMajax=[mean(majax) min(majax) median(majax) max(majax)];
minSize=curMajax(2)+2*abs(curMajax(1)-curMajax(3));
% maxSize=curMajax(4)-curMajax(1)+minSize;
% smaller=min(curMajax(1),curMajax(3))
larger=max(curMajax(1),curMajax(3));
% predSize=[smaller-curMajax(2) larger+curMajax(2)];
maxSize=larger+curMajax(2);
predSize=[minSize maxSize];

if show
    figure; imagesc(img);
    figure; imagesc(bin);
    disp(['predSize: ' num2str(predSize)]);
end

end