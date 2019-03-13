function outImg = postProcessKaggle(inImg, minSize, conn)

% fill holes
inImg = removeObjectWithinObject(inImg);

% join objects
inImg = mergeToucingObjects(inImg, conn);

% remove small
outImg = removeSmallObjects(inImg, minSize);