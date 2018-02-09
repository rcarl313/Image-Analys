function probmap = sliding_window(img, w, w0)
% Takes an image and filters it with an weigthed image w
%This filterd image is then 

filterdimg = imfilter(img, w, 'symmetric');

y = sum(filterdimg,3)+w0;

probmap = 1./(1+exp(-y)); %The sigmoid function