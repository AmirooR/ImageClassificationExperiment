%Loading train and test data
feaDir = 'F:\Rahimi\PascalVOCLLCCode\1\';
annotationDir = 'D:\Rahimi\VOCdevkit\VOC2012\ImageSets\Main\';
thisName = 'car';
featureSize = 21504; %size of final feature vector
[trainNames,~] = textread([annotationDir sprintf('%s_train.txt',thisName)],'%s %d');
[testNames,~] = textread([annotationDir sprintf('%s_val.txt',thisName)],'%s %d');
SVM_C = 100;

trainPyramid = zeros(featureSize, size(trainNames,1) );
testPyramid = zeros(featureSize, size(testNames,1) );
disp('Loading training data');
for i = 1:size(trainNames,1)
    X = load([feaDir trainNames{i} '.mat']);
    trainPyramid(:,i) = X.fea;
end

disp('Loading test data');
for i = 1:size(testNames,1)
    X = load([feaDir testNames{i} '.mat']);
    testPyramid(:,i) = X.fea;
end

class_names = {'aeroplane','bicycle','bird','boat','bottle','bus','chair','cat','cow','diningtable','dog','horse','motorbike','pottedplant','sheep','sofa','tvmonitor','car','person','train'};

for i=1:size(class_names,2)
   thisName = class_names{i};
   disp(thisName);
   [~,trainClass] = textread([annotationDir sprintf('%s_train.txt',thisName)],'%s %d');
   [~,testClass] = textread([annotationDir sprintf('%s_val.txt',thisName)],'%s %d');
   idx = 1:size(trainClass,1);
   posIdx = idx( trainClass == 1);
   negIdx = idx( trainClass == -1);
   trainIdx = [posIdx negIdx];
   trainIdx = trainIdx( randperm(size(trainIdx,2) ) );
   
   idx = 1:size(testClass,1);
   posIdx = idx( testClass == 1);
   negIdx = idx( testClass == -1);
   testIdx = [posIdx negIdx];
   
   [w, bias] = trainLinearSVM(trainPyramid(:,trainIdx),  trainClass( trainIdx ), SVM_C) ;
   scores = w' * testPyramid(:,testIdx) + bias ;
   
   % Visualize the precision-recall curve
   figure(1) ; clf ; set(1,'name','Precision-recall on train data') ;
   vl_pr(testClass(testIdx), scores) ;
   saveas(1,sprintf('results\%s_LLC_PR_VLSVM.png',thisName));
   save(sprintf('results\%s_LLC_VLSVM_MODEL.mat',thisName),'w','bias');
end
