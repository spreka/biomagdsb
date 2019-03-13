function runLengthEncodeFolder( folder, outputFolder)
%creates a submission file into the specified directory based on the mask
%images located in that directory

f = fopen(fullfile(outputFolder,'submission.csv'),'w');
fprintf(f,'ImageId,EncodedPixels\n');

d = dir(fullfile(folder,'*.png'));

for i=1:length(d)
    [~,imgID,~] = fileparts(d(i).name);
    img = imread(fullfile(folder,d(i).name));
    nuclei = runLengthEncoding(img);
    if isempty(nuclei)
        disp(d(i).name);
    end
    for j=1:length(nuclei)
        fprintf(f,'%s,',imgID);        
        for k=1:length(nuclei{j})/2
            fprintf(f,'%d %d ',nuclei{j}([2*k-1 2*k]));
        end
        fprintf(f,'\n');
    end   
end

fclose(f);

end

