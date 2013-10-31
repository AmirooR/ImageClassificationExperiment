function D = learnDictionary(dataDir, numBases, numSample)
    fileNames = dir(dataDir);
    fileNames(randperm(size(fileNames, 1))) = fileNames;
    samples = zeros(128, numSample);
    curNumSamples = 0;
    while(true)
        for ii = 1:length(fileNames)
            fprintf('current sample size = %d \n', curNumSamples); 
            subname = fileNames(ii).name;
            if ~strcmp(subname, '.') & ~strcmp(subname, '..')
                subname = fullfile(dataDir, subname);
                tmp = load(subname);
                denseSift = tmp.feaSet.feaArr;
                numSift = size(denseSift, 2);
                sampleSize = 100;
                if(curNumSamples + sampleSize > numSample)
                    sampleSize = numSample - curNumSamples;
                end
                sampleIndx = randperm(numSift, sampleSize);
                samples(:, (curNumSamples + 1):(curNumSamples + sampleSize)) = denseSift(:, sampleIndx);
                curNumSamples = curNumSamples + sampleSize;
                
                if(curNumSamples >= numSample)
                    D = vl_kmeans(samples, numBases, 'algorithm','elkan');
                    return;
                end
            end
        end
    end
end