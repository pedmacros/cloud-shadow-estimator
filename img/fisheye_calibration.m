images = imageDatastore('calibration_images')
[imagePoints,boardSize] = detectCheckerboardPoints(images.Files);
squareSize = 24; % millimeters
worldPoints = generateCheckerboardPoints(boardSize,squareSize);
I = readimage(images,27);
imageSize = [size(I,1) size(I,2)];
params = estimateFisheyeParameters(imagePoints,worldPoints,imageSize);
J1 = undistortFisheyeImage(I,params.Intrinsics);
figure
imshowpair(I,J1,'montage')
title('Original Image (left) vs. Corrected Image (right)')
