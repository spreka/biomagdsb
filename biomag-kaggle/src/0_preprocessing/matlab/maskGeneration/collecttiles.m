function collecttiles( rawImagesDir, fromDir, dstDir, ext )
    % Collect predictions
    % Collect images
    % ... for the mask generation and style transfer.
    
    % raw images: 1.png, 2.png, 3.png
    % inpDir: 1_qwer.png, 1_asdf.png, 2_123.png, 2_ioj.png, 3_xvcb.png,
    % 3_89h.png
    
    % outputDir: 1/_qwer.png, 1/_asdf.png, 2/_123.png, 2/_ioj.png, ...


    lst = dir(fullfile(rawImagesDir, ['*', ext]));
    
    for srcId=1:numel(lst)
        fName = lst(srcId).name;
        [~, fNameWExt, ~] = fileparts(fName);
        disp(['Collecting: ', fNameWExt]);
        
        srcFiles = dir(fullfile(fromDir, [fNameWExt, '*', ext]));
        for srcFileId=1:numel(srcFiles)
            srcFName = srcFiles(srcFileId).name;
            srcFilePath = fullfile(fromDir, srcFName);
            dstFileDir = fullfile(dstDir, fNameWExt);
            dstFilePath = fullfile(dstFileDir, srcFName);
            disp(['mkdir ', dstFileDir]);
            disp(['cp: ', srcFilePath, ' ', dstFilePath]);
            mkdir(dstFileDir);
            copyfile(srcFilePath, dstFilePath);
        end
    end
    
end

