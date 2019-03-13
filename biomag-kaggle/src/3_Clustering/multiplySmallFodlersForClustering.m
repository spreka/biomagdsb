function multiplySmallFodlersForClustering(inFolder, stylesToTrain, minToAction,ext)
%it leaves the merged folder untouched
%if stylesToTrain is empty then the cluster.csv is not affected

if ~isempty(stylesToTrain)
    trainT = readtable(stylesToTrain);
else
    trainT = '';
end

dirList = dir(inFolder);

for i=3:numel(dirList)
    
    fileList = dir([inFolder filesep dirList(i).name filesep]);
    if numel(fileList)-2 <= minToAction % minus 2 is to get rid of . and .. because of ambiguous extentions
        disp(['-----> Small number of files in folder: ' inFolder dirList(i).name ' -- Split started.']);
        for j=3:numel(fileList)
            if fileList(j).isdir
                continue;
            end
            in = imread([inFolder filesep dirList(i).name filesep fileList(j).name]);
            
            [sx, sy, sz] = size(in);
            
            out1 = in(1:round(sx/2), 1:round(sy/2), :);
            out2 = in(1:round(sx/2), round(sy/2):sy, :);
            out3 = in(round(sx/2):sx, 1:round(sy/2), :);
            out4 = in(round(sx/2):sx, round(sy/2):sy, :);
            
            [~,exex,~] = fileparts(fileList(j).name);
            imwrite(out1, [inFolder filesep dirList(i).name filesep exex, '_01.' ext]);
            imwrite(out2, [inFolder filesep dirList(i).name filesep exex, '_02.' ext]);
            imwrite(out3, [inFolder filesep dirList(i).name filesep exex, '_03.' ext]);
            imwrite(out4, [inFolder filesep dirList(i).name filesep exex, '_04.' ext]);                        
            
            %% Abel here
            %search for this image in the table
            if ~isempty(trainT)
                warning off            
                 for k=1:length(trainT.Name)
                    if strcmp(trainT.Name{k},fileList(j).name)
                        clusterIdx = trainT(k,2);
                        trainT(k,:) = []; %delete the entry
                        %add new ones
                        trainT.Name{end+1} = [exex, '_01.' ext];
                        trainT(end,2) = clusterIdx;
                        trainT.Name{end+1} = [exex, '_02.' ext];
                        trainT(end,2) = clusterIdx;                    
                        trainT.Name{end+1} = [exex, '_03.' ext];
                        trainT(end,2) = clusterIdx;
                        trainT.Name{end+1} = [exex, '_04.' ext];
                        trainT(end,2) = clusterIdx;
                        break;
                    end
                 end
                warning on
            end
                        
            delete([inFolder filesep dirList(i).name filesep fileList(j).name]);
            %change in merged folder
            %{
            delete([mergedFolder filesep fileList(j).name]);
            imwrite(out1, [mergedFolder filesep fileList(j).name(1:end-4), '_01.' ext]);
            imwrite(out2, [mergedFolder filesep fileList(j).name(1:end-4), '_02.' ext]);
            imwrite(out3, [mergedFolder filesep fileList(j).name(1:end-4), '_03.' ext]);
            imwrite(out4, [mergedFolder filesep fileList(j).name(1:end-4), '_04.' ext]);                        
            %}
        end
        disp('Done...');
    end
    
end

if ~isempty(trainT)
    writetable(trainT,stylesToTrain);
end
