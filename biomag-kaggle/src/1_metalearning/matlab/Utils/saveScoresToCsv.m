function saveScoresToCsv(scores,pathToScores, headers,filename)
%saveScoresToCsv save down all the scores to a csv file in pathToScores
%folder named mergedScores.csv. headers is optional

if nargin<4
    filename = 'mergedScores.csv';
end
f = fopen(fullfile(pathToScores,filename),'w');

if nargin>2    
    fprintf(f,'ImageName');
    for i=1:length(headers)
        fprintf(f,',%s',headers{i});
    end
    fprintf(f,'\n');
end

for iN = scores.keys
    imageName = iN{1};
    fprintf(f,'%s',imageName);
    currentScores = scores(imageName);
    for i=1:length(currentScores)
        fprintf(f,',%f',currentScores(i));
    end
    fprintf(f,'\n');
end

fprintf(f,'MEAN');
m = mean(cell2mat(scores.values),2);
for i=1:length(m)
    fprintf(f,',%f',m(i));
end
fprintf(f,'\n');

fclose(f);

end

