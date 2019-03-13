%script to sort the generated files to groups

inputDir = '/home/biomag/szkabel/20180321_2/180330_MORE';
targetDir = '/home/biomag/szkabel/180330_MORE';
csvFile = '/home/biomag/szkabel/styles.csv';

d = dir(fullfile(inputDir,'*.png'));
T = readtable(csvFile);

for i=1:numel(d)
    [~,exEx,~] = fileparts(d(i).name);
    ss = strsplit(exEx,'_');
    stem = ss{1};
    for j=1:length(T.Style)
        if strcmp([stem '.png'],T.Name{j})
            target = T.Style(j);
            tcd = fullfile(targetDir,['group_' num2str(target,'%d')],'test');
            if ~isdir(tcd), mkdir(tcd); end            
            copyfile(fullfile(inputDir,d(i).name),fullfile(tcd,d(i).name));
            break;
        end
    end
end