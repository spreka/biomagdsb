function [ nuclei ] = generate_n_object( N, representativeFeatureVectors, objectShape, objectTexture )
%GENERATE_N_OBJECT Generates n objects having shape features similar to
%given feature vectors.

MAX_ITER = 100;

nuclei = [];
nucleiInd = 0;

fprintf('Generating objects...\n');
iterNum = 0;
progress = 0;
while nucleiInd < N
    
    if nucleiInd/N*10>progress
        progress = progress+1;
        fprintf('#');
    end
    
    selectedCellId = randi(size(representativeFeatureVectors,1));
    reprFeatureVector = representativeFeatureVectors(selectedCellId,:);
    %the cell size is sampled from normal distribution and with maximum
    %0.2*mean of standard deviation (or the real standard dev)
    meanA = mean(representativeFeatureVectors(:,1));
    stdA = std(representativeFeatureVectors(:,1));
    desiredArea = normrnd(meanA,min(stdA,meanA*0.2));
    
    % generate an object
    n = nucleus([0 0], nucleiInd+1, sqrt(desiredArea/pi), objectShape,...
        objectTexture);
    
    shape = getshape(n);
    shapeSize = size(shape);
    
    % rescale object
    
    shape = imresize(shape, [shapeSize(1)/sqrt(reprFeatureVector(2)), shapeSize(2)*sqrt(reprFeatureVector(2))]);
    
    rotAngle = randi(360);
    shape = imrotate(im2double(shape), rotAngle, 'bilinear')>0.5;
    
    emptyRows = ~any(shape,2);
    shape(emptyRows,:) = [];
    
    emptyCols = ~any(shape,1);
    shape(:,emptyCols) = [];
    
%     object = shape;
    
    n2 = cellobj(getcoords(n), nucleiInd+1, shape, shape, struct('coords',getcoords(n), 'area',sum(shape(:))));
    
    rightFeatures = 1;
    
    % check rest of the features
    p = regionprops(getshape(n2),'Area','MinorAxisLength','MajorAxisLength','Perimeter','Solidity','Eccentricity');
    
    if length(p)==1

        if abs(p.Area/(p.Perimeter.^2) - reprFeatureVector(3)) > 0.1*reprFeatureVector(3)
            rightFeatures = 0;
        end

        if abs(p.Eccentricity - reprFeatureVector(4)) > 0.1*reprFeatureVector(4)
            rightFeatures = 0;
        end

        if abs(p.Solidity - reprFeatureVector(5)) > 0.1*reprFeatureVector(5)
            rightFeatures = 0;
        end
    else
        rightFeatures = 0;
    end
    
    if rightFeatures
        nuclei = [nuclei n];
        nucleiInd = nucleiInd + 1;
        iterNum = 0;
    else
        iterNum = iterNum+1;
    end
    
    if iterNum > MAX_ITER
        warning('Only %d objects were generated.', nucleiInd);
        break;
    end
    
end

end

