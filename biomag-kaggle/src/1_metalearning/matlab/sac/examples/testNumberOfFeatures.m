clear all;

% load data set
d=sacReadArff('c:\projects\ETH_LMC\doks\journal_ThomasWild_SmallScreen\fig_conf\HK2.arff');

% list of features
dFeatureList = [17,1,3,14,2,23,13,18,22,4,15,24,12,16,21,20,6,25,10,19,11,5,9,7,8];


for i=1:24

    pFeatureList = sort(dFeatureList(26-i:25), 'descend');
    dNew = d;
   
    for j=1:length(pFeatureList)
        dNew.featureNames(pFeatureList(j)) = [];
        dNew.featureTypes(pFeatureList(j)) = [];
        dNew.instances(:,pFeatureList(j)) = [];
    end;

    [bestOptionString, bestOptionStruct, bestE(i)] = sacSuggest('SVM_LibSVM', d, 60, 'random');    
    
end;