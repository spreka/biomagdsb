function generateValidationCustom(matlab_scripts,inFolder)
% Generates validationNames.mat for the selected directory inFolder to the
% output directory [matlab_scripts]/generateValidation/validationFileNames.mat

output=fullfile(matlab_scripts,'generateValidation/validationFileNames.mat');

% list image files in folder
exts={'png','tiff','tif','bmp','jpg','jpeg'};
l=[];
for e=1:numel(exts)
    l=[l;dir(fullfile(inFolder,'images',['*.' exts{e}]))];
end

validationNames={l(:).name}';
tmp1=cellfun(@(x) strsplit(x,'.'),validationNames,'UniformOutput',false);
validationNames=cellfun(@(y) y{1},tmp1,'UniformOutput',false);

fprintf('listed %d images\n',numel(validationNames));

% write the listed image names to file for training script to load
save(output,'validationNames');

end