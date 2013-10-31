class_names = {'aeroplane','bicycle','bird','boat','bottle','bus','chair','cat','cow','diningtable','dog','horse','motorbike','pottedplant','sheep','sofa','tvmonitor','car','person','train'};
annotation_root = '/Users/amirrahimi/Public/VOC2012/ImageSets/Main';
SVM_C = 100;
for i=1:size(class_names,2)
   thisName = class_names{i};
   disp(thisName);
   [~,trainClass] = textread([annotation_root sprintf('/%s_train.txt',thisName)],'%s %d');
   [~,testClass] = textread([annotation_root sprintf('/%s_val.txt',thisName)],'%s %d');
   
   idx = 1:size(trainClass,1);
   posIdx = idx(trainClass == 1);
   negIdx = idx(trainClass == -1);   
   %negIdx = negIdx( randperm(size(negIdx,2)) );
   %negIdx = negIdx(1:size(posIdx,2));
   trainIdx = [posIdx negIdx];
   trainIdx = trainIdx( randperm(size(trainIdx,2)) );
   
   idx = 1:size(testClass,1);
   testIdx = idx( testClass ~= 0);
   
   
   
   %trainClass( trainClass == -1 ) = -1;
   %trainClass( trainClass == 0 ) = -1;
   %testClass( testClass == 0 ) = -1;
   %testClass( testClass == -1 ) = -1;
   
   [w,bias] = trainLinearSVM(pyramid_train(trainIdx,:)', trainClass(trainIdx)', SVM_C);   
   scores = w' * pyramid_val(testIdx,:)' + bias;
   figure(1) ; clf ; set(1,'name','Precision-recall on test data') ;
   vl_pr(testClass(testIdx), scores);
   saveas(1,sprintf('data/%s_PR_LinearSVM_Pyr.png',thisName));
   save( sprintf('data/%s_Model.mat',thisName),'w','bias');
end

disp('Done!');