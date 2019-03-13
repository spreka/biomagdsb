% Input: a directory in kaggle format

dataset = '/home/biomag/etasnadi/splits/splits-kaggle-format-sep/non-histo';
datasetOut = '/home/biomag/etasnadi/splits/splits-kaggle-format-sep/non-histo-2ch';
ext = '.png';

imgs = dir(dataset);
for imgId = 1:numel(imgs)
    if imgs(imgId).isdir && ~strcmp(imgs(imgId).name, '.') && ~strcmp(imgs(imgId).name, '..')
        imgName = imgs(imgId).name;
        imgPath = fullfile(dataset, imgName, 'images', [imgName, ext]);
        imgData = imread(imgPath);
        % it is only an 1 channel image...
        if(length(size(imgData)) == 2)
            disp(['1 channel: ', imgName]);
            imOut = uint8(zeros([size(imgData), 3]));
            for ch = 1:3
                imOut(:,:,ch) = imgData;
            end
            imgOutDir = fullfile(datasetOut, imgName, 'images');
            mkdir(imgOutDir);
            imgPathOut = fullfile(imgOutDir, [imgName, ext]);
            imwrite(imOut, imgPathOut);
            disp(['Saving to: ', imgPathOut]);
        end
    end
end