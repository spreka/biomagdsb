function s = sacEval2String(E,metrics)

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

% if metrics are not requested, we decide what to put in the string
if ~exist('metrics', 'var')
    fields = fieldnames(E);
    fields = setdiff(fields, {'classNames', 'classWeights', 'dataset', 'numClasses', 'numInstances', 'predictionTime', 'trainingTime'});
    metrics = {};
    for i = 1:length(fields)
        if ~isempty(E.(fields{i}))
            metrics{end+1} = fields{i};
        end
    end
end


s = sprintf('\n');
s = [s sprintf('========== Classification Summary on %s ==========\n\n', E.dataset)];

% % display the class weights
% s = [s sprintf('Class weights\n')];
% for n = 1:E.numClasses
%     s = [s sprintf('%1.4f  \t %s\n',  E.classWeights(n), E.classNames{n})];
% end
% s = [s sprintf('\n')];

for i = 1:length(metrics)
    switch metrics{i}
        case 'ACC'
            s = [s sprintf('ACCURACY\n')];
            s = [s sprintf('%2.2f%%   \t Classification Accuracy\n\n', 100*E.ACC)];
        case 'AUC'
            s = [s sprintf('AUC     \t Area under the ROC curve\n')];
            for n = 1:E.numClasses
                s = [s sprintf('%1.4f  \t %s\n',  E.AUC(n), E.classNames{n})];
            end
            w = (E.Wauc) / sum(E.Wauc);
            s = [s sprintf('%1.4f  \t Class Weighted AUC\n\n', w * E.AUC')]; %#ok<*AGROW>
        case 'BEP'
            s = [s sprintf('BEP     \t The break-even point on PR curve\n')];
            for n = 1:E.numClasses
                s = [s sprintf('%1.4f  \t %s\n',  E.BEP(n), E.classNames{n})];
            end
            w = (E.Wbep) / sum(E.Wbep);
            s = [s sprintf('%1.4f  \t Class Weighted BEP\n\n', w * E.BEP')];
        case 'CONFUSION'
            s = [s sprintf('CONFUSION (row: true class, col: predicted class)\n')];
            ndigits = ceil(max(max(log10(E.CONFUSION))));
            abc = 'a':char(97+E.numClasses-1);
            a = cell(E.numClasses,E.numClasses); %spc = cell(1,E.numClasses);
            abclabel = cell(1,E.numClasses);
            
            for m=1:E.numClasses
                abc = [];
                for n = 1:ndigits
                    abc = [abc ' '];
                end
                abc = [abc char(97 + m -1)];
                abclabel{1,m} = abc;
            end
            for m=1:E.numClasses
                for m2 =1:E.numClasses
                    b = blanks(ndigits+1);
                    numstr = sprintf('%d', E.CONFUSION(m,m2));
                    b(1:length(numstr)) = numstr;
                    a{m,m2} = strjust(b, 'right');
                end
            end
            
            S = [];
            for m=1:E.numClasses
                S = [S abclabel{1,m} ' '];
            end
            s = [s sprintf('%s  <-- predicted to be:\n', S)];
            
            for m=1:E.numClasses
                S = [];
                for n=1:E.numClasses
                    S = [S a{m,n} ' '];
                end
                s = [s sprintf('%s | %s=%s\n',S, strtrim(abclabel{1,m}), E.classNames{m}) ];
            end
            s = [s sprintf('\n')];
        case 'CORRECT'
            s = [s sprintf('%d/%d   \t CORRECT classified instances\n\n',E.CORRECT,E.numInstances)];
        case 'ERR'
            s = [s sprintf('ERR\n')];
            s = [s sprintf('%2.2f%%   \t Classification error rate\n\n', 100*E.ERR)];
        case 'FPR'
            s = [s sprintf('FPR     \t False Positive Rate\n')];
            for n = 1:E.numClasses
                s = [s sprintf('%1.4f  \t %s\n',  E.FPR(n), E.classNames{n})];
            end
            w = (E.Wfpr) / sum(E.Wfpr);  
            s = [s sprintf('%1.4f  \t Class Weighted False Postive Rate\n\n', w * E.FPR')];
        case 'FSC'
            s = [s sprintf('FSC       \t F-score\n')];
            for n = 1:E.numClasses
                s = [s sprintf('%1.4f  \t %s\n',  E.FSC(n), E.classNames{n})];
            end
            w = (E.Wfsc) / sum(E.Wfsc);
            s = [s sprintf('%1.4f  \t Class Weighted F-score\n\n', w * E.FSC')];
        case 'INCORRECT'
            s = [s sprintf('%d/%d   \t INCORRECT classified instances\n\n',E.INCORRECT,E.numInstances)];
        case 'LFT'
            s = [s sprintf('LFT     \t Lift\n')];
            for n = 1:E.numClasses
                s = [s sprintf('%1.4f  \t %s\n',  E.LFT(n), E.classNames{n})];
            end
            w = (E.Wlft) / sum(E.Wlft);
            s = [s sprintf('%1.4f  \t Class Weighted Lift\n\n', w * E.LFT')];
        case 'MXE'
            s = [s sprintf('MXE\n')];
            s = [s sprintf('%0.4g    \t Mean cross entropy\n\n', E.MXE)];
        case 'PR'
            % do nothing; PR values are used to form a precision-recall
            % curve. BEP can be used to summarize
        case 'PRE'
            s = [s sprintf('PRE     \t Precision\n')];
            for n = 1:E.numClasses
                s = [s sprintf('%1.4f  \t %s\n',  E.PRE(n), E.classNames{n})];
            end
            w = (E.Wpre) / sum(E.Wpre);
            s = [s sprintf('%1.4f  \t Class Weighted Precision\n\n', w * E.PRE')];
        case 'REC'
            s = [s sprintf('REC     \t Recall\n')];
            for n = 1:E.numClasses
                s = [s sprintf('%1.4f  \t %s\n',  E.REC(n), E.classNames{n})];
            end
            w = (E.Wrec) / sum(E.Wrec);
            s = [s sprintf('%1.4f  \t Class Weighted Recall\n\n', w * E.REC')];
        case 'RMS'
            s = [s sprintf('RMS\n')];
            s = [s sprintf('%1.4f  \t Root mean squared error\n\n', E.RMS)];
        case 'ROC'
            % do nothing; ROC values are used to make an ROC curve. AUC can
            % be used to summarize
        case 'TPR'
            s = [s sprintf('TPR     \t True Positive Rate\n')];
            for n = 1:E.numClasses
                s = [s sprintf('%1.4f  \t %s\n',  E.TPR(n), E.classNames{n})];
            end
            w = (E.Wtpr) / sum(E.Wtpr);
            s = [s sprintf('%1.4f  \t Class Weighted True Postive Rate\n\n', w * E.TPR')];
    end
end