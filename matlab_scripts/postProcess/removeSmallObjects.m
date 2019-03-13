function inImg = removeSmallObjects(inImg, size)

stats = regionprops(inImg, 'Area');

for i=1:length(stats)
    if stats(i).Area <= size
        pix = find(inImg == i);
        inImg(pix) = 0;
    end
end