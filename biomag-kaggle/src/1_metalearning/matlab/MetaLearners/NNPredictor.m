classdef NNPredictor < MetaLearner
    %NNPREDICTOR Neural networks implementation for the predictor
    %interface.
    %   Neural networks are efficient learning algorithms they are
    %   widespread. They are capable to classification and regression too,
    %   but it cannot predict distributions for regression. (In regression
    %   they are only points in the space predicted)
    
    properties
        model
        numHiddenNeurons;
        trainRatio;
        testRatio;
        valRatio;
        transferFunction;
    end
    
    methods
        function obj = NNPredictor(origDir,scores)
            obj.origDir = origDir;
            obj.scores = scores;    
            fprintf('You have instantiated a new Neural Network predictor\n');
                        
            numHiddenNeurons = 30;                        
            trainRatio = 0.7;                        
            valRatio = 0.2;            
            testRatio = 0.1;                                                               
                        
            obj.numHiddenNeurons = numHiddenNeurons;
            obj.trainRatio = trainRatio;
            obj.testRatio = testRatio;
            obj.valRatio = valRatio;            
            obj.transferFunction = 'logsig';
        end
        
        function obj = train(obj)
            [features,~,imageNames] = loadBasicFeatures(obj.origDir,0,1);
            
            inputs = cell2mat(features)';
            targets = cell2mat(obj.scores.values);
            disp('Train with Neural Networks');
            obj.model = newfit(inputs,targets,obj.numHiddenNeurons);

            obj.model.divideParam.trainRatio = obj.trainRatio;  
            obj.model.divideParam.valRatio = obj.valRatio;
            obj.model.divideParam.testRatio = obj.testRatio;

            obj.model.layers{1}.transferFcn = obj.transferFunction;

            % Trying to make the IW (inputWeights) determined 
            %(because it changed it's size without this)
            %net.inputs{1}.size = size(inputs,1); % It is necessary for the adopting methods.
            %net.inputs{1}.processFcns = {'mapminmax'};
            %net.outputs{2}.processFcns ={};

            % Train and Apply Network
            obj.model.trainParam.showWindow = false;
            obj.model = train(obj.model,inputs,targets);           
        end
        
        function [predictedScores,out,imageNames] = predict(obj,testDir)
            
             [features,~,imageNames] = loadBasicFeatures(testDir,0,1);
             inputs = cell2mat(features)';
             disp('Predict with Neural Networks');
             outputs = sim(obj.model,inputs);             
             
             predictedScores = containers.Map;
             for i=1:length(imageNames)
                 predictedScores(imageNames{i}) = outputs(:,i);
                 [~,out(i)] = max(outputs(:,i));
             end             
        end
        
        function print2file(obj,f)
            fprintf(f,'MathWorks Neural Network Predictor.\n');
            fprintf(f,['It is capable to predict distributions: ' num2str(obj.isDistribution) '\n']);
            fprintf(f,'------\n');
            fprintf(f,['Number of hidden neurons: ' num2str(obj.numHiddenNeurons) '\n']);
            fprintf(f,['Train ratio: ' num2str(obj.trainRatio) '\n']);
            fprintf(f,['Test ratio: ' num2str(obj.testRatio) '\n']);
            fprintf(f,['Validation ratio: ' num2str(obj.valRatio) '\n']);
            fprintf(f,['The used transfer function is: ' obj.transferFunction '\n']);
        end
    end
    
end

