function [outimg, className, mapping] = createClassImage(clusterFolder)
% AUTHOR:	Lassi Paavolainen, Tamas Balassa
% DATE: 	April 22, 2016
% NAME: 	createClassImage
% REDESIGNED by szkabel 2018
% 
% To create the table-like image containing the "icon images" of a class of
% annotated cells.
%
% INPUT:
%   counter         Identification number assigned to the class.
%
% OUTPUT:
%   outimg          Contain the table for containing the "icon images".
%   className       Name of the class.
%   mapping         ImageName
%
% COPYRIGHT    

    imgsize = [512 512];
    sepsize = 10;
    cols = 7;
    width = cols * imgsize(1) + (cols+1) * sepsize;
    
    S = strsplit(clusterFolder,filesep);   
    className = S{end};
    d = dir(fullfile(clusterFolder,'*.png'));
    N = length(d);
    classCells = cell(1,N);    
    for i=1:N
        [classCells{i},maps] = imread(fullfile(clusterFolder,d(i).name));        
        if size(classCells{i},3) ~= 3
            if size(maps,2)==1 || isempty(maps)
                classCells{i} = repmat(classCells{i},1,1,3);            
            else
                classCells{i} = ind2rgb(classCells{i},maps);                
            end
        end
    end    
    
    numimgs = size(classCells,2);
    rows = ceil(numimgs / cols);
    height = rows * imgsize(2) + (rows+1) * sepsize;
    outimg = uint8(zeros(height,width,3));
    outimg(:,:,:) = 50;
    
    mapping = {};
    x = 1;
    y = 1;
    for i = 1:numimgs       
        classCells{i} = im2uint8(classCells{i});
        resImg = uint8(zeros(imgsize(1),imgsize(2),3));
        resImg(:) = 50;
        origImgSize = size(classCells{i});
        if origImgSize(1)>origImgSize(2)
            tmpImg = imresize(classCells{i}, [imgsize(1) NaN]);
            resImg(1:size(tmpImg,1),1:size(tmpImg,2),:) = tmpImg;
        else
            tmpImg = imresize(classCells{i}, [NaN imgsize(2)]);
            resImg(1:size(tmpImg,1),1:size(tmpImg,2),:) = tmpImg;
        end
        if x > cols
           x = 1;
           y = y + 1;
        end
        offsetw = imgsize(1) * (x-1) + sepsize * x;
        offseth = imgsize(2) * (y-1) + sepsize * y;
        outimg(offseth:offseth+imgsize(2)-1,offsetw:offsetw+imgsize(1)-1,:) = resImg;        
        subimgmeta.ImageName = d(i).name;        
        mapping{y,x} = subimgmeta;
        x = x + 1;
    end
end