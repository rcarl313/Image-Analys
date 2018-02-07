function net = train_classifier(layers, imgs_train, labels_train, imgs_val, labels_val)

options = trainingOptions('sgdm', ...
            'LearnRateSchedule', 'piecewise', ...
            'LearnRateDropFactor', 0.2, ...
            'LearnRateDropPeriod', 5, ...
            'MaxEpochs', 20, ...
            'MiniBatchSize', 300);

net = trainNetwork(imgs_train, labels_train, layers, options);

     truelabel = net.classify(imgs_val) == labels_val;

nbrcorr = (sum(truelabel)/length(imgs_val))*100;

disp([num2str(nbrcorr) '% of the validation numbers where classified correctly'])

