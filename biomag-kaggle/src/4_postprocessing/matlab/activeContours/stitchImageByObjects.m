function failed=stitchImageByObjects(src,name,pad,dest)
% Stitches an output image from small 40-px padded images.
% src: folder of small images to stitch together as
% [src]\[name]\out\[name*.png]
% name: name of the original image
% pad: padding size in pixels
% dest: folder to write output stitched image
% bboxFile: .mat file from splitImagesByObjects.m for this image

if nargin<4
    dest=fullfile(src,name,'outStitched\');
end

if ~exist(dest,'dir')
    mkdir(dest);
end

[~,name,~]=fileparts(name);
% l=dir(fullfile(src,[name '*.png']));
% b=load(bboxFile);
if ~exist(fullfile(src,[name '_bbox.mat']),'file')
    failed=true;
    return;
end
b=load(fullfile(src,[name '_bbox.mat']));
n=size(b.bboxSize,1);
out=zeros(b.s); % size of original image
missCount=0;
failed=false;
for i=1:n
%     imName=fullfile(src,name,'out',sprintf('%s_%03d.png.png',name,i));    % file name was too long!
    imName=fullfile(src,name,'out',sprintf('%03d.png.png',i));

    if ~exist(imName,'file')       
        fprintf('file %s does not exist\n',imName);
        missCount=missCount+1;
        imName=fullfile(src,name,'masks',sprintf('%03d.png',i));
        part=imread(imName);
        %       if missCount>ceil(n/10)
%             failed=true;
%             return;
%         end
      %  continue;
    else
        padded=imread(imName);
        part=double(padded(pad+1:end-pad,pad+1:end-pad,3)/intmax(class(padded))); % mask is on blue channel
    end
        
    % remove 40-by-40 px padding
   
    part=imfill(part,'holes');
    bbox=ceil(b.bboxSize(i,:));
%   newS=[max(1,bbox(1)+b.p) max(1,bbox(2)+b.p) min(b.s(1),bbox(3)-b.p) min(b.s(2),bbox(4)-b.p)];

    % handle overlapping objects created by individual contour fitting
    tmp=out(bbox(2):min(bbox(2)+bbox(4),b.s(1)),bbox(1):min(bbox(1)+bbox(3),b.s(2)));
    % always keep the 1st found object
    part(tmp~=0&part~=0)=0;
    out(bbox(2):min(bbox(2)+bbox(4),b.s(1)),bbox(1):min(bbox(1)+bbox(3),b.s(2)))=tmp+double(i*part);
end
imwrite(uint16(out),fullfile(dest,[name '.png']));