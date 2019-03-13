function runfromKaggleToClusters(mergedImagesDir,clusterDir,clusteringType,initialSegmentation,sacFolder,failCounter,canContinue)

counter=1;
done=false;
while counter<=4 && ~done
	done=fromKaggleToClusters(mergedImagesDir,clusterDir,clusteringType,initialSegmentation,sacFolder,failCounter,canContinue);
	
	if ~done
		% create missing data folder
		if ~exist(fullfile(sacFolder,'data'))
			fprintf('-creating data folder-\n');
			mkdir(fullfile(sacFolder,'data'));
		else
			fprintf('-deleting data folder-\n');
			rmdir(fullfile(sacFolder,'data'));
		end
		%counter=counter+1;
	end
end
