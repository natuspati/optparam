clearvars

%% separate images
cd testimgs5

fileList = dir('*.tif');

for i=1:length(fileList)
    img = imread(fileList(i).name);
    left_img = img(:,1:2016,:);
    left_img_gray = rgb2gray(left_img);
    right_img = img(:,2017:end,:);
    right_img_gray = rgb2gray(right_img);
    imwrite(left_img_gray, strcat('left_imgs/left', string(i),'.tif'));
    imwrite(right_img_gray, strcat('right_imgs/right', string(i),'.tif'));
end

cd ..