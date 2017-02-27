
load('data/sbdd/dataset/cls/2010_000001.mat')
image = imread('data/sbdd/dataset/img/2010_000001.jpg');

imshow(image); figure;
imshow(GTcls.Segmentation*255)