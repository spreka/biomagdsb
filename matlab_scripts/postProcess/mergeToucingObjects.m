function inImg = mergeToucingObjects(inImg, conn)

% create neighbour matrix
nMtx = neighbourhoodMatrix(inImg, conn);

change = 1;

while change
    change = 0;
    mergeList = [];
    for i=1:size(nMtx)
        for j=i:size(nMtx)
            
            if nMtx(i, j) == 1
                % object1
                o1 = inImg==i;
                tmpI1 = inImg*0;
                tmpI1(o1) = 1;
                stat1 = regionprops(tmpI1, 'Area', 'ConvexArea');                
                % object2
                o2 = inImg==j;
                tmpI2 = inImg*0;
                tmpI2(o2) = 1;
                stat2 = regionprops(tmpI2, 'Area', 'ConvexArea');                
                %merge
                tmpI3  = (tmpI1+tmpI2)>0;
                stat3 = regionprops(tmpI3, 'Area', 'ConvexArea');                            
                
                if ((stat1(1).ConvexArea - stat1(1).Area) + ...
                (stat2(1).ConvexArea - stat2(1).Area)) > stat3(1).ConvexArea - stat3(1).Area
                    disp(['Merge: ' num2str(i) ' and ' num2str(j)]);
                    mergeList(end+1, 1) = i;
                    mergeList(end, 2) = j;
%                     inImg(o1) = j;
%                     inImg(o2) = j;
                    change = 1;
                end                
            end
        end
    end
    if change
        for i=1:size(mergeList, 1)
            o1 = find(inImg == mergeList(i, 1));
            inImg(o1) = mergeList(i, 2);
        end
        inImg = relabel(inImg);            
        nMtx = neighbourhoodMatrix(inImg, conn);
    end
end

inImg = relabel(inImg);
