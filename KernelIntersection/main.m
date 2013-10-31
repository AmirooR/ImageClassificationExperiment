% Compute Intersection Kernel on training/testing data into K/KK matrices ( I used code provided by Prof. Lazebnik for this task )
% Create a folder to push the results
% Link to libsvm

class_names = {'aeroplane','bicycle','bird','boat','bottle','bus','chair','cat','cow','diningtable','dog','horse','motorbike','pottedplant','sheep','sofa','tvmonitor','car','person','train'};
totalACC = 0;
for i=1:size(class_names,2)
   thisName = class_names{i};
   disp(thisName);
   [~,trainClass] = textread(sprintf('D:\\Rahimi\\VOCdevkit\\VOC2012\\ImageSets\\Main\\%s_train.txt',thisName),'%s %d');
   [~,testClass] = textread(sprintf('D:\\Rahimi\\VOCdevkit\\VOC2012\\ImageSets\\Main\\%s_val.txt',thisName),'%s %d');
   idx = 1:size(trainClass,1);
   goodIdx = idx( trainClass ~= 0);
   trainIdx = goodIdx( randperm(size(goodIdx,2) ) );
   
   idx = 1:size(testClass,1);
   testIdx = idx( testClass ~= 0);
   KTrain = [ (1:size(trainIdx,2))',K(trainIdx,trainIdx)];
   KTest = [ (1:size(testIdx,2))',KK(testIdx,trainIdx)];
   model = svmtrain(trainClass(trainIdx),KTrain,'-t 4');
   [predClass,acc,decVals] = svmpredict(testClass(testIdx),KTest,model);
   save(sprintf('results/%s_model_IntersectionKernel.mat',thisName),'model','trainClass','testClass','predClass','acc','decVals');
   disp(acc);
   confusionmat(testClass(testIdx),predClass)
   totalACC = totalACC + acc;
   figure(1) ; clf ; set(1,'name','Precision-recall on test data') ;
   vl_pr(testClass(testIdx), decVals) ;
   saveas(1,sprintf('results\\%s_Intersection_PR_LIBSVM.png',thisName));
end

totalACC = totalACC/size(class_names,2);
disp(totalACC);

