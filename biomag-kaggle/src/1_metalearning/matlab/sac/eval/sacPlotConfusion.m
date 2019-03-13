function sacPlotConfusion(CONFUSION, classNames)
% sacPlotConfusion
% Plots the confusion matrix to evaluate the
% quality of a classifier.
%
% INPUT:
%
% 
%
% OUTPUT:           
%
% See also: sacAccuracy, sacConfusion, sacPlotROC

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

%y = sacLabels2All(y);

%plotconfusion(y', p');


N = size(CONFUSION,1);

if ~exist('classNames', 'var');
    for i = 1:N
        classNames{i} = sprintf('Class %d', i);
    end
end
    

I = zeros([N N 3]);
    
figure;
hold on;

set(gcf, 'Color', [1 1 1]);

BoxSize = 100;

for r = 1:N
    %row2 = N-r+1;
    row = N-r+1;    
    
    for c = 1:N
        col = c;
        %col2 = c+1;
        
        totalNumClass = sum(CONFUSION(r,:));
        totalInstances = sum(sum(CONFUSION));
        
        % if diagonal
        if r == c
            correctPercent = CONFUSION(r,c) / totalNumClass;
            %I(row,col,:) = getColor(correctPercent, 2);
            color = getColor(correctPercent, 2);
        else
            incorrectPercent = CONFUSION(r,c) / totalNumClass;
            %I(row,col,:) = getColor(incorrectPercent, 1);
            color = getColor(incorrectPercent, 1);
        end
        
        xpts = [col-.5 col+.5 col+.5 col-.5 col-.5];
        ypts = [row-.5 row-.5 row+.5 row+.5 row-.5];
        
        patch(xpts,ypts,color);
        
        
        dataPercent = (CONFUSION(r,c) / totalInstances) * 100;
        
        numText = sprintf('%d', CONFUSION(r,c));
        percentText = sprintf('(%2.1f%%)', dataPercent);
        
        text(col,row,numText, 'FontSize', 14, 'HorizontalAlignment', 'center');
        text(col,row-.25, percentText, 'FontSize', 8, 'HorizontalAlignment', 'center');
        
    end
end

midx = N/2 + .5;
midy = N/2 + .5;

text(midx, N +.75, 'Predicted Class', 'FontSize', 16, 'HorizontalAlignment', 'center');
text(N + .75, midy, 'Actual Class', 'FontSize', 16, 'HorizontalAlignment', 'center','Rotation', -90);

axis equal;
set(gca, 'XColor', [1 1 1]);
set(gca, 'YColor', [1 1 1]);


for i = 1:N
   row = N-i+1;  
   col = i;
   text(.40,row,classNames{i}, 'FontSize', 8, 'HorizontalAlignment', 'center','Rotation', 90,'Interpreter', 'none'); 
   text(col,.35,classNames{i}, 'FontSize', 8, 'HorizontalAlignment', 'center','Interpreter', 'none');  
end


hold off;



function color = getColor(val,chan)
% value should be between 0 and 1

if chan == 1
% should be red when value is high
r = .6 + val;
g = .6 - val;
b = .6 - val;

else
% should be green when value is high
r = .5 - .2*val;
g = .5 + .5*val;
b = .5 - .2*val;    
    
    
end

color = [r g b];

