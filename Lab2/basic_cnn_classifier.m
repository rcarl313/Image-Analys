function layers = basic_cnn_classifier()

layers = [
imageInputLayer([28 28 1]);
convolution2dLayer(5, 20);%Create a 
%convolutional layer with 10 filters that have a height and width of 28, 
%and use a stride of 2 in the horizontal and vertical directions.
reluLayer();
maxPooling2dLayer(2,'Stride', 2);
fullyConnectedLayer(10);
softmaxLayer();
classificationLayer()];