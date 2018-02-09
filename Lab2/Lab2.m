%% 2.1
load cell_examples.mat

%% 2.2

cell = examples{1,1};% loads example with label 1
notcell = examples{1,11671};% loads example with label 0

figure(1)
imagesc(cell)
figure(2)
imagesc(notcell)

%%



% [wgrad, w0grad] = partial_gradient(w, w0, example, label)
% 
% scal = dot(example,w);
% y = sum(scal(:))+w0; %calculates y = I*w +w0, we get a scalar
% %p = exp(y)/(1+exp(y)) = 1/(1/exp(y))+1 = 1/(exp(-y)+1) 
% %calculates probability with sigmoid 
% p = 1/(exp(-y)+1);
% 
% 
% % Calculate the partial gradient of w and w0, w0grad*Image = wgrad
% %L_i = -ln(p), p = 1/(e^-y + 1), y = dot(I,w) + w0
% %dy/dw = I where I is an image
% % from chain rule we get dL_i/dw =(p-1)*I for positive imagaes and
% %dL_i/dw=p*I for negative images
% %dL_i/dw=(p-1)
% w0grad = (p - label);
% wgrad = example * w0grad;

%% 2.4

w = rand(29,29,3);
w0 = zeros(29,29,3);
lrate=0.5;
example = examples{1,1};
labels = labels(1);

[w, w0] = process_epoch(w, w0, lrate, example, labels);

%% 2.5
load cell_examples.mat
examplesrand=randperm(length(examples));
w = 0.01 * randn(29,29,3);
w0 = 0;
lrate=0.01;

for i=1:13000
    ii=examplesrand(i);
example = examples{1,ii};
label = labels(ii);

[w, w0] = process_epoch(w, w0, lrate, example, label);

end

imagesc(w)

%% 2.7

img= read_image(['cells-test/042.png']); %Reads a test image
probmap = sliding_window(img, w, w0); %Calls the made function sliding_window
show_with_overlay(img, probmap>0.5) %Uses the given function to show an image with the probability of a cell

%% 2.8

threshold = 0.5; %Sets a threshold for comming function
gaussian_std = 3; %Sets standard deviation to 3

maxima = strict_local_maxima(probmap, threshold, gaussian_std); %Runs the created function strict_local_maxima

imagesc(img); colormap gray; axis image; %Show an image in grayscale
hold on;
plot(maxima(1,:), maxima(2,:),'r*') %Plots the maximas as red stars on the image

%% 2.9

[imgs, labels] = digitTrain4DArrayData; %Loads training data for the network

imagesc(imgs(:,:,:,342)), axis image, colormap gray %Shows an image from the data

%% 2.10
%see the function basic_cnn_classifier

%% 2.11

options = trainingOptions('sgdm'); %Creates training options for the network
layers = basic_cnn_classifier(); %Creates a basic cnn classifier with different layers
net = trainNetwork(imgs, labels, layers, options); %Trains a network with te previous options and layers

%% 2.12

%Tests the network with the following commands
img=imgs(:,:,:,1);

net.predict(img)
net.classify(img)

%% 2.13
%how many trainable parameters does your network contain?
conv_param = ((5*5*1+1)*20); %Trainable parameters in the convolution layer
fully_param = (12*12*20+1)*10; %Trainable parameters in the fully-connected layer

tot_param = conv_param+fully_param; %Trainable parameters
% (previous_filter_size*nbrOf_previous_feature_maps+1)*feature_maps

%% 2.14
%4/5 as training and use the last 1/5 for validation av trained cnn
clc, clear all, close all

[imgs, labels] = digitTrain4DArrayData; %Loads in train images for training of better_cnn

layers = basic_cnn_classifier(); %Creates the layers
randimgs = randperm(length(imgs)); %Randomizes the images
imgs_train = imgs(:,:,:,randimgs(1:4000)); %Selects the first 4000 digits for training
labels_train = labels(randimgs(1:4000)); %-||- labels
imgs_val = imgs(:,:,:,randimgs(4001:5000)); %Selects the last 1000 digits for validation
labels_val = labels(randimgs(4001:5000));% -||- labels
k=1; %Initial value for k
net = train_classifier(layers, imgs_train,... %Runs the function train_classifier
    labels_train, imgs_val, labels_val,k)
for k =2:10 %Continues to rin the function for additional epochs
net = train_classifier(net.Layers, imgs_train, labels_train, imgs_val, labels_val,k) %Loops through epochs of validation
end

%% 2.15

input = 28*28*1; %Size of input image
conv1 = 28*28*10; %Size after first convolution
conv2 = 28*28*10; %Blue
max1 = 20*20*10; %size after maxpool
conv3 = 20*20*10;
max2 = 10*10*10;
conv4 = 10*10*20 %Red

%The Blue convolution takes 3.92 times as long to compute due to the
%reductions with two maxpool.

%% 2.16 
clc, clear all, close all

[imgs, labels] = digitTrain4DArrayData; %Loads in train images for training of better_cnn

layers = better_cnn_classifier(); %Creates the layers
randimgs = randperm(length(imgs)); %Randomizes the images
imgs_train = imgs(:,:,:,randimgs(1:4000)); %Selects the first 4000 digits for training
labels_train = labels(randimgs(1:4000)); %-||- labels
imgs_val = imgs(:,:,:,randimgs(4001:5000)); %Selects the last 1000 digits for validation
labels_val = labels(randimgs(4001:5000));% -||- labels
k=1; %Initial value for k
net = train_classifier(layers, imgs_train,... %Runs the function train_classifier
    labels_train, imgs_val, labels_val,k)
for k =2:10 %Continues to rin the function for additional epochs
net = train_classifier(net.Layers, imgs_train, labels_train, imgs_val, labels_val,k) %Loops through epochs of validation
end

%% 2.17 ,2.18

[imgs_test, labels_test] = digitTest4DArrayData; %Loads 5000 images with digits for validation of network

truelabel2 = net.classify(imgs_test) == labels_test;    %Finds the correctly classified digits

nbrcorr1 = (sum(truelabel2)/length(imgs_test))*100; %Percentage of correctly classified digits

disp([num2str(nbrcorr1) '% of the test numbers where classified correctly']) %Displays the percentage of correctly classified digits

wrong = find(truelabel2 == 0)   %Finds all the wrongly classified digits


for i =1:10
u = wrong(i,1);    %Finds the i:th image that was wrongly classified
wrongnr = net.classify(imgs_test(:,:,:,u)); %Picks out which number the image has
k = char(wrongnr); %Converts the categorial int into a charachter
imwrite(imgs_test(:,:,:,u), (['mistaken_as_' k '.png'])) %Saves the image with a name whoch include which digit it was mistaken for
end

