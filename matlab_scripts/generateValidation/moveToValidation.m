function moveToValidation(inFolder, validationFolder, validationFileNames)
% 
% % copy data into validation folder
% inFolder = 'c:/projects_tmp/NucleiCompetition/data/ourAnnotation/clean/AllDataStandardStage2_UNET/';
% validationFolder = 'c:/projects_tmp/NucleiCompetition/data/ourAnnotation/clean/validationStage2_UNET/';

mkdir(validationFolder);
mkdir([validationFolder filesep 'images']);
mkdir([validationFolder filesep 'masks']);


load(validationFileNames);

for i=1:numel(validationNames)
    
    fileList = dir(fullfile(inFolder,'images',[validationNames{i} '*.png']));    
    for j=1:numel(fileList)
         [PATHSTR,NAME,EXT] = fileparts([inFolder '/images/' fileList(j).name]);
        if length(NAME) == 64
            movefile([inFolder '/images/' fileList(j).name ], [validationFolder '/images/' fileList(j).name]);
        else
            delete([inFolder '/images/' fileList(j).name]);
        end
    end

    
    fileList = dir([inFolder '/masks/' validationNames{i} '*.tiff']);
    for j=1:numel(fileList)
         [PATHSTR,NAME,EXT] = fileparts([inFolder '/masks/' fileList(j).name]);
        if length(NAME) == 64        
            movefile([inFolder '/masks/' fileList(j).name ], [validationFolder '/masks/' fileList(j).name]);
        else
            delete([inFolder '/masks/' fileList(j).name]);
        end
    end       
end

