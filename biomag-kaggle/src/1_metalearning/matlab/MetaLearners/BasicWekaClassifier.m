classdef BasicWekaClassifier < MetaLearner
    %BasicShallowNet
    %   Implements a porting to weka through ACC and sac
    
    properties       
        
        CommonHandles        
        %To store the trained model in sac
        wekaName
        %Name of the sac function to train
    end
    
    methods
        function obj = BasicWekaClassifier(origDir,scores,wekaName)                                                                                
            obj.origDir = origDir;
            obj.scores = scores;    
            if isnumeric(wekaName)
                list = sacGetSupportedList('classifiers');
                wekaName = list{wekaName};               
            end
            fprintf('You have instantiated a new Weka %s classifier.\n',wekaName);
            obj.wekaName = wekaName;
        end
                
        function train(obj)
            [features,~,imageNames] = loadBasicFeatures(obj.origDir,0,1);
            CH = mapFeaturesToCommonHandles( features,obj.scores, imageNames);
            CH.ClassifierNames = {obj.wekaName};
            CH.SelectedClassifier = 1;
            try
                CH = trainClassifier(CH);
            catch
                load config.mat;
                sacInit(fullfile(codeBase,'1_metalearning','matlab','sac'));
                CH = trainClassifier(CH);
            end 
            
            obj.CommonHandles = CH;
        end
                    
        function [predictedScores,out,imageNames] = predict(obj,testDir)
            [features,~,imageNames] = loadBasicFeatures(testDir,0,1);
            features = cell2mat(features);
            try
                [out,probs] = predictClassifier(features,obj.CommonHandles);
            catch
                load config.mat;
                sacInit(fullfile(codeBase,'1_metalearning','matlab','sac'));
                [out,probs] = predictClassifier(features,obj.CommonHandles);
            end             
            predictedScores = containers.Map;
            for i=1:length(imageNames)
                predictedScores(imageNames{i}) = probs(i,:);
            end
        end
        
        function performance(obj) 
            training_model_properties_GUI(obj.CommonHandles);
        end
    end
    
end

