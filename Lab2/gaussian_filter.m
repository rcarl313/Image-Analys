function result = gaussian_filter(img, sigma)
%Makes a gaussian filter for a specific image.
sizex = size(img,2);    %Amount of collumns in the image
sizey = size(img,1);    %Amount of rows in the image
hsize = ([sizey sizex]);    %Puts the number of collumns and rows in an array

h = fspecial('gaussian', hsize, sigma); %Creates a gaussian filter of the same size as the image
result = imfilter(img, h, 'symmetric'); %Filters the image with the created filter




