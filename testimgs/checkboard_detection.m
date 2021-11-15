
%% 0.Clear all
clc;
clear all;
close all;

%% 1.Read images
[Filename, Pathname]=uigetfile({'*_0.bmp'},'load images'); 
path_imgLeft=strcat(Pathname,'*_0.bmp');    
dir_imgLeft=dir(path_imgLeft);

for i=1:1%length(dir_imgLeft)
    Image_name=dir_imgLeft(i).name;
    [imagePoints,boardSize,imagesUsed] = detectCheckerboardPoints(Image_name);    
    Img=imread(Image_name);
         
    if i<10
        mLName=strcat('mL0',num2str(i),'.mat');
    elseif i>9
        mLName=strcat('mL',num2str(i),'.mat');
    end
    
    mL=imagePoints';
%     save(mLName,'mL')
    writematrix(mL, 'leftpoints.csv')
    
%     figure;
%     imshow(Img);
%     hold on;
%     plot(imagePoints(:,1),imagePoints(:,2),'ro');
end


path_imgRight=strcat(Pathname,'*_1.bmp');    
dir_imgRight=dir(path_imgRight);

for i=1:length(dir_imgRight)
    Image_name=dir_imgRight(i).name;
    [imagePoints,boardSize,imagesUsed] = detectCheckerboardPoints(Image_name);    
    Img=imread(Image_name);
    
    
    if i<10
        mRName=strcat('mR0',num2str(i),'.mat');
    elseif i>9
        mRName=strcat('mR',num2str(i),'.mat');
    end
    
    mR=imagePoints';
%     save(mRName,'mR')
    writematrix(mR, 'rightpoints.csv')
    
%     figure;
%     imshow(Img);
%     hold on;
%     plot(imagePoints(:,1),imagePoints(:,2),'ro');
end

%% 3D coordinates in target coordinate system
% M=zeros(3,size(imagePoints,1));
% 
% [Y_left1,Z_left1]=meshgrid(-64:8:-8,48:-8:0);
% Y_left1=Y_left1';
% Z_left1=Z_left1';
% M(2,1:size(Y_left1,1)*size(Y_left1,2))=reshape(Y_left1,1,size(Y_left1,1)*size(Y_left1,2));
% M(3,1:size(Z_left1,1)*size(Z_left1,2))=reshape(Z_left1,1,size(Z_left1,1)*size(Z_left1,2));
% 
% [X_left2,Y_left2]=meshgrid(8:8:56,-64:8:-8);
% % X_left2=X_left2';
% % Y_left2=Y_left2';
% M(1,size(X_left2,1)*size(X_left2,2)+1:end)=reshape(X_left2,1,size(X_left2,1)*size(X_left2,2));
% M(2,size(Y_left2,1)*size(Y_left2,2)+1:end)=reshape(Y_left2,1,size(Y_left2,1)*size(Y_left2,2));
% 
% % figure;
% % scatter3(M(1,:),M(2,:),M(3,:),'.','c');
% save('M.mat','M')




