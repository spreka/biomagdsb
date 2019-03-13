function convertdirtiff(inpDir, outDir)
    pngs = dir(fullfile(inpDir, '*.png'));
    for pngId=1:numel(pngs)
        pngName = pngs(pngId).name;
        pngPath = fullfile(inpDir, pngName);
        [~, tiffName, ~] = fileparts(pngName); 
        tiffPath = fullfile(outDir, [tiffName '.tiff']);
        disp(['Png path: ' pngPath ' tiff path: ' tiffPath]);
        im = imread(pngPath);
        imwrite(im, tiffPath);
    end
end