function ensembleProbabilies(probMapMainFolder)

p = gcp('nocreate');
if isempty(p)
    parpool('local',feature('numcores'));
end

ensembledFolder = fullfile(probMapMainFolder,'ensembled');
if ~exist(ensembledFolder,'dir')
    mkdir(ensembledFolder);
else
    delete(fullfile(ensembledFolder,'*'));
end
d = dir(probMapMainFolder);
d(ismember({d.name},{'.','..','ensembled'})) = [];
d(~cat(1,d.isdir)) = [];

if ~isempty(d)
    imagesList = dir(fullfile(probMapMainFolder, d(1).name, 'validation', '*.png'));
    if ~isempty(imagesList)
        parfor i=1:length(imagesList)
            fprintf('[%5d] Ensembling probmaps for image %s\n',i,imagesList(i).name);
            for j=1:length(d)
                if j==1
                    ensembled = double(imread(fullfile(probMapMainFolder, d(j).name, 'validation', imagesList(i).name)));
                else
                    ensembled = ensembled + double(imread(fullfile(probMapMainFolder, d(j).name, 'validation', imagesList(i).name)));
                end
            end
            ensembled = ensembled / length(d);
            imwrite(uint16(ensembled), fullfile(ensembledFolder, imagesList(i).name), 'BitDepth', 16);
        end
    end
end
