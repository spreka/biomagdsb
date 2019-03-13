function resaveAsPng(folder,extention)
%resaves all images in the folder as png files.
%if the image is 1 by 1 then tries to replace it with an appropriately
%sized zero image

%it saves to the input folder

d = dir([folder filesep '*.' extention]);

for i=1:numel(d)
    img = imread(fullfile(folder,d(i).name));
    [~,imgName,~] = fileparts(d(i).name);
    if size(img,1) == 1 && size(img,2) == 1
        if exist('config.mat','file')
            load config.mat;
            if exist(fullfile(testCheckDir,[imgName '.png']),'file')
                origImg = imread(fullfile(testCheckDir,[imgName '.png']));
                img = zeros(size(origImg));
            elseif exist(fullfile(groundTruthDir,[imgName '.png']),'file')
                origImg = imread(fullfile(groundTruthDir,[imgName '.png']));
                img = zeros(size(origImg));
            end            
        end
    end
    
    imwrite(img,fullfile(folder,[imgName '.png']));
end

end