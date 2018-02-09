function layers = basic_cnn_classifier()
%Defines a basic cnn classifier that was given in the Lab-PM
layers = [
imageInputLayer([28 28 1]);
convolution2dLayer(5, 20);
reluLayer();
maxPooling2dLayer(2,'Stride', 2);
fullyConnectedLayer(10);
softmaxLayer();
classificationLayer()];
