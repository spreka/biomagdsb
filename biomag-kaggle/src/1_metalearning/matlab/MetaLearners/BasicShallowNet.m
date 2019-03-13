classdef BasicWekaClassifier < MetaLearner
    %BasicShallowNet
    %   Implements a shallow network to identify best pipeline for an image
    
    properties
        numHiddenNeurons;
        testRatio;
        trainRatio;
        valRatio;
        transferFunction;             
        
        CommonHandles
        %To store the trained model in sac
    end
    
    methods
        function obj = BasicWekaClassifier(origDir,scores,numHiddenNeurons,trainRatio,valRatio,testRatio,transFunc)
            fprintf('You have instantiated a new Neural Network predictor\n');
                                    
            % fill in values to ensure that NN also has 'default' constructor
            if nargin<3
                numHiddenNeurons = 30;
            end
            if nargin <4
                trainRatio = 0.7;
            end
            if nargin <5
                valRatio = 0.2;
            end
            if nargin < 6
                testRatio = 0.1;                                                   
            end
            if nargin < 7
                transFunc = 'logisg';
            end
                     
            obj.origDir = origDir;
            obj.scores = scores;
            obj.numHiddenNeurons = numHiddenNeurons;
            obj.trainRatio = trainRatio;
            obj.testRatio = testRatio;
            obj.valRatio = valRatio;            
            obj.transferFunction = transFunc;            
        end
                
        function train(obj)
            [features,~,imageNames] = loadBasicFeatures(obj.origDir,0,1);
            CH = mapFeaturesToCommonHandles( features,scores, imageNames);
            CH.ClassifierNames = ''
            CH.ClassifierNames
            try
                CH = trainClassifier(CH);
            catch
                load config.mat;
                sacInit(fullfile(codeBase,'1_metalearning','matlag','sac'));
                CH = trainClassifier(CH);
            end 
            
            obj.CommonHandles = CH;
        end
                    
        function predictedScores = predict(obj,testDir)
            [features,~,imageNames] = loadBasicFeatures(obj.origDir,0,1);
            
        end
    end
    
end

