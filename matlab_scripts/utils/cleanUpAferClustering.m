function cleanUpAferClustering(inFolderMain)
% moves clustering additional files to separate folder

l=[dir(fullfile(inFolderMain,'outputs','images','*.mat')); dir(fullfile(inFolderMain,'outputs','images','*.csv'))];
if numel(l)~=3
	fprintf('additional files found in %s, exiting clean-up\n',inFolderMain);
else
	mkdir(fullfile(inFolderMain,'outputs','clustering_misc'));
	for i=1:3
		movefile(fullfile(inFolderMain,'outputs','images',l(i).name),fullfile(inFolderMain,'outputs','clustering_misc',l(i).name));
	end
end

end
