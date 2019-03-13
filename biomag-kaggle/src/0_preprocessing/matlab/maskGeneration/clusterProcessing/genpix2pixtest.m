function genpix2pixtest(inpDir, outDir, ext)
    % generates the Pix2pix test set:
    % Input: generated-mask-directory/$IMAGE_ID/$IMAGE_ID_x_y$EXT
    
    conts = dir(inpDir);
    for imId=1:numel(conts)
        imName = conts(imId).name;
        
        if strcmp(imName, '.') || strcmp(imName, '..'), continue; end
        
        disp(['Processing: ' imName]);
        % Collect generated masks
        
        contGenMasks = dir(fullfile(inpDir, imName, ['*' ext]));
        mDstDir = fullfile(outDir, imName, 'test');
        if exist(mDstDir, 'dir')~=7, mkdir(mDstDir); disp(['mkdir' mDstDir]); end
        for mId=1:numel(contGenMasks)
            mName = contGenMasks(mId).name;
            mPath = fullfile(inpDir, imName, mName);
            mDstPath = fullfile(mDstDir, mName);
            

            copyfile(mPath, mDstPath);
            disp(['cp ' mPath ' ' mDstPath]);
        end
    end
end