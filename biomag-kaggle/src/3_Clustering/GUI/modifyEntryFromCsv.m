function  modifyEntryFromCsv( csvFile, imageNameList,value )
%cluster csv removal
%   sets entries to -1 to indicate that it is not in a cluster

T = readtable(csvFile);

for i=1:length(imageNameList)
    for j=1:numel(T)
        if strcmp(imageNameList{i},T.Name{j})
            T.Style(j) = value;
            break;
        end
    end
end

writetable(T,csvFile);

end

