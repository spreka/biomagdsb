function [ filePath ] = generateSimpleOptions( targetDir, nofSplits )

optionT = table();

optionT.SplitName{1,1} = '0';
optionT.Weight(1,1) = 1;
warning off
for i=2:nofSplits    
    optionT.SplitName{end+1,1} = num2str(i-1,'%d'); % we are counting from 0
    optionT.Weight(end,1) = 1;
end
warning on
filePath = fullfile(targetDir,['basicOptions_' num2str(nofSplits,'%02d') '.csv']);
writetable(optionT,filePath);

end

