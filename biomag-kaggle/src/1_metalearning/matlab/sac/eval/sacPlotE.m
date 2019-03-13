function sacPlotE(E, plotType)
% Plots the reciever operating characteristic (ROC), confusion matrix, or
% precision-recall curve (PR) evaluating the performance of a classifier on
% a particular dataset

% From the Suggest a Classifier Library (SAC), a Matlab Toolbox for cell
% classification in high content screening. http://www.cellclassifier.org/
% Copyright © 2011 Kevin Smith and Peter Horvath, Light Microscopy Centre 
% (LMC), Swiss Federal Institute of Technology Zurich (ETHZ), Switzerland. 
% All rights reserved.
%
% This program is free software; you can redistribute it and/or modify it 
% under the terms of the GNU General Public License version 2 (or higher) 
% as published by the Free Software Foundation. This program is 
% distributed WITHOUT ANY WARRANTY; without even the implied warranty of 
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU 
% General Public License for more details.


numClasses = numel(E.classNames);
map = .75*jet(numClasses);

switch upper(plotType)
    case 'ROC'
        hold on;
        for c = 1:numClasses
            plot(E.ROC.fpr{c}, E.ROC.tpr{c}, 'Color', map(c,:));
        end
        legend(E.classNames, 'Location', 'SouthEast');
        for c = 1:numClasses
            plot(E.FPR(c), E.TPR(c), 'ko', 'Color', map(c,:), 'MarkerEdgeColor', [.3 .3 .3], 'MarkerFaceColor', map(c,:), 'MarkerSize', 6);
        end
        grid on;
        xlabel('False Positive Rate');
        ylabel('True Positive Rate');
        title(E.dataset);
        hold off;
        
    case 'PR'
        hold on;
        numClasses = numel(E.classNames);
        for c = 1:numClasses
            plot(E.PR.rec{c}, E.PR.pre{c}, 'Color', map(c,:));
        end
        legend(E.classNames, 'Location', 'SouthWest');
        xlabel('Recall');
        ylabel('Precision');
        title(E.dataset);
        
        for c = 1:numClasses
            plot(E.REC(c), E.PRE(c), 'ko', 'Color', map(c,:), 'MarkerEdgeColor', [.3 .3 .3], 'MarkerFaceColor', map(c,:), 'MarkerSize', 6);
        end
        for c = 1:numClasses
            plot(E.BEP(c), E.BEP(c), 'k*', 'Color', map(c,:), 'MarkerSize', 6);
        end
        grid on;
        hold off;
        
    case 'CONFUSION'
        
end

