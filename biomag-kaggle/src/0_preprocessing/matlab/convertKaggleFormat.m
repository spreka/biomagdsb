% Converts a set of images into the Kaggle format:
% inpDir contents: 1.png, 2.png, 3.png
% outDir contents will be:
% 1/images/1.png, 2/images/2.png, 3/images/3.png
function convertKaggleFormat(inpDir, outDir, ext)
    conts = dir([inpDir, '/*', ext]);
    nFiles = numel(conts);
    for fid = 1:nFiles
        fName = conts(fid).name;
        [~, fNameWext, ~] = fileparts(fName);
        targetDir = fullfile(outDir, fNameWext, 'images');
        srcPath = fullfile(inpDir, fName);
        dstPath = fullfile(targetDir, fName);
        % make the outdir/{image_id}/images directory
        disp(['mkdir ', targetDir]);
        mkdir(targetDir);
        disp(['cp ', srcPath, ' ', dstPath]);
        copyfile(srcPath, dstPath);
    end
end