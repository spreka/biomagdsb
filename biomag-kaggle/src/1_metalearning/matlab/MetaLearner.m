classdef (Abstract) MetaLearner < handle
    %MetaLearner 
    %   To learn for an input image which pipeline is the best for that
    %   specific image.
    %   The constructor must fill in the properties listed below.
    
    properties        
        origDir
        %The directory for the original images
        scores
        %The structure that stores the scores     
    end
    
    methods (Abstract)
        train(obj);
        %trains the classifier to give a score for the pipes
        %occasionally, if a learner works also on the mask images, then
        %train can have an optional parameter, the directory for the masks.
        
        predictedScores = predict(obj,testDir)
        %predicts scores for all images in the trainDir folder.
        %   OUTPUT:
        %       predictedScores: a map keys are imageNames of the
        %       testDirectory. Values are matrices with length of P where P
        %       is the number of pipes listed in the scores values (the length of a single value in scores)
        
    end
    
end

