function [ROC AUC PR BEP] = sacRankMetrics(L,P)
% Computes rank-based measures such as ROC, AUC, PR, and BEP

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



% the number of classes
numClasses = size(P,2);


% convert the labels vector into a binary representation
L = sacLabels2All(L);



% compute the ROC and PR stats for each class vs. rest-of-classes
for m = 1:numClasses
    clear pre rec tpr fpr pdescend ldescend A;

    % sort the probabilities and associated labels (descending values)
    [pdescend, inds] = sort(P(:,m), 'descend');
    ldescend = L(inds,m);
    
    % determine the number of total true positive and negative instances
    % positive corresponds to class m, negative to all other classes
    POS = sum(ldescend == 1);
    NEG = sum(ldescend == 0);

    % initialize variables
    TP = 0;             % true positives
    FP = 0;             % false positives
    FN = POS;           % false negatives
    TN = NEG;           % true negatives
    TPprev = 0;         % previous true positive count
    FPprev = 0;         % previous false positive count
    A = 0;              % area under the ROC 
    pDLast = -Inf;      % the previous probability value
    
    n = 1;              % index into the data points on the ROC and PR curve
    
    % loop through all of the sorted probabillities
    for i = 1:length(pdescend)

        % if probability i is not the same value as the last, compute the
        % various metrics (tpr, fpr, pre, rec, A)
        if pdescend(i) ~= pDLast

            tpr(n) = TP / POS; %#ok<AGROW>          True positive rate
            fpr(n) = FP / NEG; %#ok<AGROW>          False positive rate
            pre(n) = TP / (TP + FP); %#ok<AGROW>    Precision
            rec(n) = TP / (TP + FN); %#ok<AGROW>    Recall
            n = n + 1;
            
            % the area under the ROC
            A = A + trapezoidArea(FPprev,FP,TPprev,TP);
            
            % current values become previous values for next iteration
            FPprev = FP;
            TPprev = TP;
            pDLast = pdescend(i);
        end
        
        % if instance i has a positive label, update TP and FN, otherwise
        % update FP and TN
        if ldescend(i) == 1
            TP = TP + 1;
            FN =  (POS - TP);
        else
            FP = FP + 1;
            TN = (NEG - FP); 
        end

        
        % sanity check (commented)
        if (TP + FP + TN + FN) ~= (POS + NEG)
            fprintf('%d + %d + %d + %d = %d\n', TP,FP,TN,FN, TP+FP+TN+FN);
            fprintf('%d + %d = %d\n', POS, NEG, POS + NEG);
            keyboard;
        end

    end

    % update the metrics one last time
    tpr(n) = TP / POS; %#ok<AGROW>
    fpr(n) = FP / NEG; %#ok<AGROW>
    pre(n) = TP / (TP + FP); %#ok<AGROW>
    rec(n) = TP / (TP + FN); %#ok<AGROW>
    
    % compute the final area under the ROC curve (normalize to 1,1 area)
    A = A + trapezoidArea(FPprev,NEG, TPprev, POS);
    A = A / (NEG * POS);

    % Correct for the saw-tooth shape of the PR curve
    % Precision-recall curves have a distinctive saw-tooth shape: 
    % if the  document retrieved is nonrelevant then recall is the 
    % same as for the top  documents, but precision has dropped. If 
    % it is relevant, then both precision and recall increase, and 
    % the curve jags up and to the right. It is often useful to remove 
    % these jiggles and the standard way to do this is with an 
    % interpolated precision: the interpolated precision   
    % at a certain recall level  is defined as the highest 
    % precision found for any recall level. From:
    % http://nlp.stanford.edu/IR-book/html/htmledition/evaluation-of-ranked-retrieval-results-1.html
    for i = length(rec):-1:1
        pre(i) = max(pre(i:end));
    end

    
    % determine the break-even point
    b = abs(pre-rec);
    bmin = min(b);
    bind = find(b == bmin, 1, 'first');
    bep = (pre(bind) + rec(bind)) / 2;
    
    

    % store the metrics for return to the calling function
    ROC.tpr{m} = tpr;
    ROC.fpr{m} = fpr;
    PR.pre{m} = pre;
    PR.rec{m} = rec;
    AUC(m) = A; %#ok<AGROW>
    BEP(m) = bep; %#ok<AGROW>
end




function A = trapezoidArea(x1,x2,y1,y2)
% computes the trapeziodal area given two base coordinates and two height
% coordinates for use in computing the AUC

base = (x2-x1);
height = (y2+y1)/2;
A = base * height;







