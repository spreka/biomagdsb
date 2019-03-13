function inImg = mergeTouchingObjects(inImg, conn)
%mergeTouchingObjects recursively merges 2 adjacent regions if the concave
% area of the merged object is smaller than the sum of the concave areas of
% the 2 objects. conn can be 4 or 8 neighbourhood.

% TODO check number of iterations of merging

fprintf('\nMerging touching objects... \n');

% create neighbour matrix
nMtx = neighbourhoodMatrix(inImg, conn);

change = 1;
counterForDisplay = 1;

while change
    change = 0;
    mergeList = [];
    
    neighboursList = find(nMtx==1);
    if ~isempty(neighboursList)
        neighboursList = reshape(neighboursList, 1, []);
        for n = neighboursList
            [i,j] = ind2sub(size(nMtx), n);
            % object1
            o1 = inImg==i;
%             tmpI1 = zeros(size(inImg));
%             tmpI1(o1) = 1;
            stat1 = regionprops(o1, 'Area', 'ConvexArea');
            % object2
            o2 = inImg==j;
%             tmpI2 = zeros(size(inImg));
%             tmpI2(o2) = 1;
            stat2 = regionprops(o2, 'Area', 'ConvexArea');
            %merge
            tmpI3  = o1 | o2;
            stat3 = regionprops(tmpI3, 'Area', 'ConvexArea');
            %                 fprintf('sol1=%0.3f, sol2=%0.3f, (sol1+sol2)/2=%0.3f, sol3=%0.3f\n',stat1.Solidity,stat2.Solidity,(stat1.Solidity+stat2.Solidity)/2,stat3.Solidity);
            %                 fprintf('val1=%0.3f, val2=%0.3f, (val1+val2)/2=%0.3f, val3=%0.3f\n',(stat1(1).ConvexArea - stat1(1).Area)/stat1(1).ConvexArea,(stat2(1).ConvexArea - stat2(1).Area)/stat2(1).ConvexArea,(stat1(1).ConvexArea - stat1(1).Area)/stat1(1).ConvexArea+(stat2(1).ConvexArea - stat2(1).Area)/stat2(1).ConvexArea,(stat3(1).ConvexArea - stat3(1).Area)/stat3(1).ConvexArea);
            
            if ((stat1(1).ConvexArea - stat1(1).Area)/stat1(1).ConvexArea + ...
                    (stat2(1).ConvexArea - stat2(1).Area)/stat2(1).ConvexArea) > (stat3(1).ConvexArea - stat3(1).Area)/stat3(1).ConvexArea
                
                fprintf('[%4d and %4d] ', i, j);
                if counterForDisplay>5
                    counterForDisplay = 1;
                    fprintf('\n');
                else
                    counterForDisplay = counterForDisplay + 1;
                end
                mergeList(end+1, 1) = i;
                mergeList(end, 2) = j;
                %                     inImg(o1) = j;
                %                     inImg(o2) = j;
                change = 1;
            end
        end
    end
    
    if change
        for i=1:size(mergeList, 1)
            inImg(inImg == mergeList(i, 1)) = mergeList(i, 2);
        end
        inImg = relabelImage(inImg);
        nMtx = neighbourhoodMatrix(inImg, conn);
    end
end

inImg = relabelImage(inImg);
