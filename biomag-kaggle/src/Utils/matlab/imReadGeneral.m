function img = imReadGeneral(filename)

[img,maps] = imread(filename);
if size(img,3) ~= 3
    if size(maps,2)==1 || isempty(maps)
        img = repmat(img,1,1,3);
    else
        img = ind2rgb(img,maps);
    end
end

end