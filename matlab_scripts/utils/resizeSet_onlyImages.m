function resizeSet_onlyImages(inImageFolder, outImageFolder, scale)
% from Peter's original resizeSet code

disp([inImageFolder ' started...']);
mkdir(outImageFolder);

% main image
inImageList = dir(fullfile(inImageFolder,'*.png'));

for imi=1:numel(inImageList)
	inImage = imread(fullfile(inImageFolder,inImageList(imi).name));
	outImage = imresize(inImage, scale, 'bicubic');
	imwrite(uint8(outImage), fullfile(outImageFolder,inImageList(imi).name));
end

disp([inImageFolder ' done...']);