
%% 0.Clear all
clc;
clear all;
close all;

%% 1.Read images
[Filename, Pathname]=uigetfile({'*.jpg'},'Read images'); 
path_img=strcat(Pathname,'*.jpg');    
dir_img=dir(path_img);

% NewPathname=strcat(Pathname,'seprated images\');
% mkdir(Pathname,'separated images'); 

for i=1:length(dir_img)
    Image_name=dir_img(i).name;
    Img=imread(Image_name);
    Img=rgb2gray(Img);    
    j=i;
    if j<10
         Img_left_name=strcat('Cal-0',num2str(j),'_0.bmp');
         Img_right_name=strcat('Cal-0',num2str(j),'_1.bmp');
         Img_full_name=strcat('Img0',num2str(j),'.bmp');
    elseif j>9
        Img_left_name=strcat('Cal-',num2str(j),'_0.bmp');
        Img_right_name=strcat('Cal-',num2str(j),'_1.bmp');
        Img_full_name=strcat('Img',num2str(j),'.bmp');
    end    
    Img_left=Img;
    Img_left(:,size(Img_left,2)/2:end)=0;
    Img_right=Img;
    Img_right(:,1:size(Img_left,2)/2)=0;
    
    
%     Img_left=Img_left(1:2:end,1:2:end);
%     Img_right=Img_right(1:2:end,1:2:end);
%     Img_full=Img(1:2:end,1:2:end);
%     Img_left_name=strcat(NewPathname,Img_left_name);  
%     Img_right_name=strcat(NewPathname,Img_right_name);

    imwrite(Img_left,Img_left_name);
    imwrite(Img_right,Img_right_name);
%     imwrite(Img_full,Img_full_name);
end



