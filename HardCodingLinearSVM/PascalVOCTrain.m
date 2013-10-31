% set pyramid_train (N * D) and pyramid_val (M * D), where N is the number
% of training instances and M is the number of testing instances.
% Setup vlfeat
% correct the paths for your own use


class_names = {'train'};
%class_names = {'aeroplane','bicycle','bird','boat','bottle','bus','chair','cat','cow','diningtable','dog','horse','motorbike','pottedplant','sheep','sofa','tvmonitor','car','person'};
annotation_root = '/Users/amirrahimi/Public/VOC2012/ImageSets/Main';
SVM_C = 100;
for i=1:size(class_names,2)
   thisName = class_names{i};
   disp(thisName);
   [~,trainClass] = textread([annotation_root sprintf('/%s_train.txt',thisName)],'%s %d');
   [~,testClass] = textread([annotation_root sprintf('/%s_val.txt',thisName)],'%s %d');
   idx = 1:size(trainClass,1);
   goodIdx = idx( trainClass ~= 0);
   trainIdx = goodIdx( randperm(size(goodIdx,2) ) );
   
   idx = 1:size(testClass,1);
   testIdx = idx( testClass ~= 0);
  
   [w,bias] = trainLinearSVM(pyramid_train(trainIdx,:)', trainClass(trainIdx)', SVM_C);   
   scores = w' * pyramid_val(testIdx,:)' + bias;
   figure(1) ; clf ; set(1,'name','Precision-recall on test data') ;
   vl_pr(testClass(testIdx), scores);
   saveas(1,sprintf('results/%s_PR_LinearSVM_Pyr.png',thisName));
   save( sprintf('results/%s_Model.mat',thisName),'w','bias');
end

disp('Done!');
