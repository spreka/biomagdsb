function extendFeaturesCsv(csvFile,features)
%extendFeatures Adds additional entries to the csvFile, it opens the file
%and writes the data in.

data = csvread(csvFile);
newData = [data; features];
csvwrite(csvFile,newData);

end

