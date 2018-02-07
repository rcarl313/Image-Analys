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

%for i=1:3
    img= read_image(['cells-test/042.png']);
    probmap = sliding_window(img, w, w0);
    show_with_overlay(img, probmap>0.5)
%end

%% 2.8

threshold = 0.5;
gaussian_std = 3;

maxima = strict_local_maxima(probmap, threshold, gaussian_std);

imagesc(img); colormap gray; axis image;
hold on;
plot(maxima(1,:), maxima(2,:),'r*')

%% 2.9

[imgs, labels] = digitTrain4DArrayData;

imagesc(imgs(:,:,:,342)), axis image, colormap gray

%% 2.10
%see the function basic_cnn_classifier

%% 2.11

options = trainingOptions('sgdm');
layers = basic_cnn_classifier();
net = trainNetwork(imgs, labels, layers, options);

%% 2.12
img=imgs(:,:,:,1);

net.predict(img)
net.classify(img)

%% 2.13
%how many trainable parameters does your network contain?

%% 2.14
%4/5 as training and use the last 1/5 for validation av trained cnn
layers = basic_cnn_classifier();
randimgs = randperm(length(imgs));
imgs_train = imgs(:,:,:,randimgs(1:4000));
labels_train = labels(randimgs(1:4000));
imgs_val = imgs(:,:,:,randimgs(4001:4940));
labels_val = labels(randimgs(4001:4940));

net = train_classifier(layers, imgs_train,... 
    labels_train, imgs_val, labels_val)

%% 2.15

%For the network below, compare (roughly) the time consumption of the blue 
%and the red.

%% 2.16 
layers = better_cnn_classifier();
randimgs = randperm(length(imgs));
imgs_train = imgs(:,:,:,randimgs(1:4000));
labels_train = labels(randimgs(1:4000));
imgs_val = imgs(:,:,:,randimgs(4001:4940));
labels_val = labels(randimgs(4001:4940));

net = train_classifier(layers, imgs_train,... 
    labels_train, imgs_val, labels_val)