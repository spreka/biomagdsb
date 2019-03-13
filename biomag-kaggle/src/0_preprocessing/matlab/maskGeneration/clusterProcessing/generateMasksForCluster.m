function generateMasksForCluster( inpDir, outDir, ext, nMasks, objType)
%GENERATEMASKS Summary of this function goes here
%   Detailed explanation goes here
%   In the inputDir, there are subdirectories. Each of them has several
%   masks. We want to generate masks based on each subdirectory's contents.
    clusters = dir(inpDir);

    extWOdot = ext;
    if strcmp(ext(1), '.')
        extWOdot = ext(2:end);
    end
        
    for clusId = 1:numel(clusters)
        clusName = clusters(clusId).name;
        if ~strcmp(clusName, '.') && ~strcmp(clusName, '..')
            clusInputDir = fullfile(inpDir, clusName);
            masksOutDir = fullfile(outDir, clusName);
            if exist(masksOutDir, 'dir') == 7 
                continue;
            end
            disp(['mkdir' masksOutDir]);
            mkdir(masksOutDir);
            disp(['Generating masks for cluster: ', clusInputDir, '->', masksOutDir]);
            maskGeneration(clusInputDir, masksOutDir, nMasks, extWOdot, objType);
        end
    end

end

function maskGeneration(inputFolder, outputFolder, nMasks, extWOdot, objType)    
    % The generation step...
    load('config.mat');
    load(cellMaskDataBase);
    gm = generateMasks(DB, inputFolder, outputFolder, extWOdot);
    if strcmp(objType,'nuclei')
        gm.generateMasksToFolderSub(1:nMasks,[512 512],Inf);
    elseif strcmp(objType,'cytoplasms')
        gm.generateMasksToFolderSubCyto(1:nMasks,[512 512],Inf);
    else
        fprintf('Unsupported object type to generate\n');
        return;
    end
end
