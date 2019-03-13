function outImg = removeSmallObjects(inImg, minimalSize)
% removeSmallObjects clears objects under pixel area size.

outImg = inImg;
stats = regionprops(inImg, 'Area');
areas = [0; cat(1,stats.Area)];
areaMap = areas(inImg+1);
outImg(areaMap<minimalSize) = 0;

outImg = relabelImage(outImg);
