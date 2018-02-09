function net = train_classifier(layers, imgs_train, labels_train, imgs_val, labels_val,k)
%Creates a network with the MATLAB-Function trainNetwork.
%The network is trained with a set of data that includes both training
%images and validation images. 


options = trainingOptions('sgdm',...        %Defines the options for the network
            'LearnRateSchedule', 'piecewise', ...
            'LearnRateDropFactor', 0.95, ...
            'LearnRateDropPeriod', 1, ...
            'MaxEpochs', 5, ...
            'InitialLearnRate', 0.2*0.95^(5*(k-1)));

net = trainNetwork(imgs_train, labels_train, layers, options); %Uses the MATLAB-function trainNetwork to train a convolutional neural network

truelabel = net.classify(imgs_val) == labels_val; %Calculates the correectly classified images

nbrcorr = (sum(truelabel)/length(imgs_val))*100; %Calculates the epercentage of correctly classifed images

disp([num2str(nbrcorr) '% of the validation numbers where classified correctly']) %Displays the percentage




