function splittedObjects = splitObjectsBySeeds(labelledObjects, seeds, varargin)
% splitObjectsBySeeds splits segmented objects of labelledObjects based on
% extra seed detected possibly via other method
%
%   Inputs:
%       labelledObjects: MxN matrix, each label value is assigned to an
%                       individual segmented object
%       seeds: MxN matrix where 0 values belong to background, positive
%                       values are assigned to object center candidates
%       splitType: 'distance' or 'watershed'
%   Output:
%       splittedObjects: MxN matrix. Objects that contain more than 1 seed
%                       points coming from other method are Voronoi
%                       tesselated
%
%   Example:
%       objects = zeros(50);
%       objects(randperm(numel(objects),10)) = 1; % add 10 object seeds
%       L = bwlabel(objects);
%       LD = imdilate(L,ones(7));
%       LDMERGED = bwlabel(LD>0); % merge adjacent objects
%       figure; imagesc(LDMERGED); colorbar; title('Merged objects');
%       SPLITTED = splitObjectsBySeeds(LDMERGED, L); %splits objects if two or more seed are inside of an object of LDMERGED
%       figure; imagesc(SPLITTED); colorbar; title('Splitted objects');

defaultSplittingType = 'watershed';
expectedSplittingTypes = {'watershed', 'distance'};
defaultOptions = {};

p = inputParser;
addRequired(p, 'labelledObjects');
addRequired(p, 'seeds');
addOptional(p, 'splitType', defaultSplittingType, @(x) any(validatestring(x,expectedSplittingTypes)));
addParameter(p, 'options', defaultOptions, @iscell);

parse(p, labelledObjects, seeds, varargin{:});

splittedObjects = labelledObjects;

uniqueValues = unique(labelledObjects);
maxValue = uniqueValues(end);

if length(uniqueValues)>1
    for labelInd = 2:length(uniqueValues)
        currentObject = labelledObjects==uniqueValues(labelInd);
        currentSeeds = bwlabel(seeds>0 & currentObject);
        if numel( find(currentSeeds>0) ) > 1
            if strcmp(p.Results.splitType,'distance')
                [~, Labels] = bwdist( currentSeeds ); 
                splittedObject = currentSeeds(Labels) .* double((currentObject>0)) + double(maxValue)+1;
                splittedObjects(currentObject) = splittedObject(currentObject);
            elseif strcmp(p.Results.splitType,'watershed')
                if isempty(p.Results.options)
                    [dist, ~] = bwdist( ~currentObject );
                    dist = imimposemin(-dist,currentSeeds>0);
                    splittedObject = watershed(dist);
                    splittedObject( ~currentObject) = 0;
                    splittedObjects(currentObject) = splittedObject(currentObject);
                else
                    error('splitObjectsBySeeds:notImplemented', 'Watershed for intensity based splitting is not implemented yet.');
                end
            end
            maxValue = maxValue+numel(find(currentSeeds>0));
        end
    end
else
    warning('Input labels image does not contain any objects.');
end
