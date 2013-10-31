%Put all images to the following folder with Train and Val as subfolders
% I put all images in the Train folder, though!
image_dir = 'D:\Rahimi\VOCdevkit\VOC2012\images';

%folder for saving sift features
sift_dir = 'F:\Rahimi\PascalVOCSift';
extr_sift(image_dir,sift_dir);

disp('Learning Dictionary');
B = learnDictionary([sift_dir '\Train'], 1024, 100000); % It seems that 100000 data points and 1024-D codes  are small. 
														% Larger values may increase the precision. However, it needs more CPU,RAM, and Running time.
														
save('dictionary\PascalVOC_SIFT100K_Kmeans_1024.mat', 'B');

disp('Learning Codes by LLC');
pyramid = [1, 2, 4];                % spatial block structure for the SPM
knn = 5;                            % number of neighbors for local coding


% -------------------------------------------------------------------------
% set path
% go to vlfeat/toolbox and run vl_setup to add vlfeat to your matlab

data_dir = sift_dir;       % directory for saving SIFT descriptors
fea_dir = 'F:\Rahimi\PascalVOCLLCCode';    % directory for saving final image features

database = retr_database_dir(data_dir);

if isempty(database),
    error('Data directory error!');
end

Bpath = ['dictionary/PascalVOC_SIFT100K_Kmeans_1024.mat'];

load(Bpath);
nCodebook = size(B, 2);              % size of the codebook

% -------------------------------------------------------------------------
% extract image features

dFea = sum(nCodebook*pyramid.^2);
nFea = length(database.path);

fdatabase = struct;
fdatabase.path = cell(nFea, 1);         % path for each image feature
fdatabase.label = zeros(nFea, 1);       % class label for each image feature

for iter1 = 1:nFea,  
    if ~mod(iter1, 5),
       fprintf('.');
    end
    if ~mod(iter1, 100),
        fprintf(' %d images processed\n', iter1);
    end
    fpath = database.path{iter1};
    flabel = database.label(iter1);
    
    load(fpath);
    [rtpath, fname] = fileparts(fpath);
    feaPath = fullfile(fea_dir, num2str(flabel), [fname '.mat']);
    
 
    fea = LLC_pooling(feaSet, B, pyramid, knn);
    label = database.label(iter1);

    if ~isdir(fullfile(fea_dir, num2str(flabel))),
        mkdir(fullfile(fea_dir, num2str(flabel)));
    end      
    save(feaPath, 'fea', 'label');

    
    fdatabase.label(iter1) = flabel;
    fdatabase.path{iter1} = feaPath;
end;


%Training and Testing with LinearSVM
disp('Train and Test with SVM');

class_names = {'aeroplane','bicycle','bird','boat','bottle','bus','chair','cat','cow','diningtable','dog','horse','motorbike','pottedplant','sheep','sofa','tvmonitor','car','person','train'};

%Loading train and test data
feaDir = 'F:\Rahimi\PascalVOCLLCCode\1\';
annotationDir = 'D:\Rahimi\VOCdevkit\VOC2012\ImageSets\Main\';
thisName = 'class_names{1}; %it is a tmp name
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

disp('Done!');
