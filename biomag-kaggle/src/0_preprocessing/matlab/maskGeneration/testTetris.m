%perform mask generation by TETRIS method to a test image

%testImg should be there and a DB database

N = 1000;

measuredFeatures = {'Area','ShapeRatio','Circularity','Eccentricity','Solidity'};

DB = FeatureDataBase;

vectors = DB.measureCellProperties(testImg,measuredFeatures);

[fID,iID,rID,~] = DB.fetchCloseIndex(measuredFeatures,vectors,N);

maskArray = DB.fetchMasks(fID,iID,rID);

