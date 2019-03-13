function [done,pairwiseDistanceMatrix,imageNames] = predictDistanceDone(CH,mergedImagesDir,sacFolder)

	done=false;

	try
		[pairwiseDistanceMatrix,imageNames] = predictDistance(CH,mergedImagesDir);
		done=true;
	catch ex
		if ~exist(fullfile(sacFolder,'data'))
			fprintf('-creating data folder-\n');
			mkdir(fullfile(sacFolder,'data'));
		else
			fprintf('-deleting data folder-\n');
			rmdir(fullfile(sacFolder,'data'));
		end
		pairwiseDistanceMatrix=[];
		imageNames=[];
	end
