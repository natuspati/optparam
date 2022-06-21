images = dir('2d/*.tiff'); 
patternDims = [4, 7];
img = imread(strcat('2d/', images(1).name));
grayimg = rgb2gray(img);
imagePoints = detectCircleGridPoints(grayimg,patternDims,PatternType="asymmetric");