function outImg = postProcessKaggle(inImg, minSize, conn)

% fill holes
inImg = removeObjectWithinObject(inImg);

% join objects
inImg = mergeTouchingObjects(inImg, conn);

% remove small
outImg = removeSmallObjects(inImg, minSize);
