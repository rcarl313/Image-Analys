function layers = better_cnn_classifier()
%Defines layers for a cnn classifier 

layers = [
imageInputLayer([28 28 1]);
convolution2dLayer(5, 30,'padding', 2);
reluLayer();
maxPooling2dLayer(2,'Stride', 2);%-->14x14x30
convolution2dLayer(3, 60, 'padding', 1);%--->14x14x*60
reluLayer
dropoutLayer();
fullyConnectedLayer(10);
softmaxLayer();
classificationLayer()];

% image is 1x28x28 --> conv 30 5x5-filters + relu --> 30x28x28 --> max pooling 
% --> 30x14x14 --> conv 60 3x3-filters + relu --> 60x14x14 