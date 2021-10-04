
clear array 
clear target
clear count
clear countStop

[dataType,samples,responseType,concentration,replicates,gasType] = size(responseContainer4);
stopCount = 1;
count = 1;
array = zeros(gasType*concentration*replicates*samples,responseType*2);
target= zeros(gasType*concentration*replicates*samples,-2);
arrayd = zeros(gasType*concentration*replicates*samples,responseType*2);
targetd= zeros(gasType*concentration*replicates*samples,gasType*concentration);

%only abs data if d = 1 & 2 are used
for d= [1]
    for a=1:responseType
        for b=1:gasType
            for c=1:concentration
                tic
                for e=1:replicates
                    for f=1:samples
                        data = responseContainer4(d,f,a,c,e,b);
                        array(count,a) = data;
                        target(count,1) = (b*5-5)+c;
                        count = count+1;
                    end
                end
                toc
            end
        end
        countStop(stopCount)= count
        count = 1;
        stopCount = stopCount+1
    end
end
for d= [2]
    for a=1:responseType
        for b=1:gasType
            for c=1:concentration
                tic
                for e=1:replicates
                    for f=1:samples
                        data = responseContainer4(d,f,a,c,e,b);
                        array(count,responseType+a) = data;
                        target(count,1) = (b*5-5)+c;
                        count = count+1;
                    end
                end
                toc
            end
        end
        countStop(stopCount)= count
        count = 1;
        stopCount = stopCount+1
    end
end
for d= [1]
    for a=1:responseType
        for b=1:gasType
            for c=1:concentration
                tic
                for e=1:replicates
                    for f=1:samples
                        data = responseContainer4(d,f,a,c,e,b);
                        arrayd(count,a) = data;
                        targetd(count,(b*5-5)+c) = 1;
                        count = count+1;
                    end
                end
                toc
            end
        end
        countStop(stopCount)= count
        count = 1;
        stopCount = stopCount+1
    end
end
for d= [2]
    for a=1:responseType
        for b=1:gasType
            for c=1:concentration
                tic
                for e=1:replicates
                    for f=1:samples
                        data = responseContainer4(d,f,a,c,e,b);
                        arrayd(count,responseType+a) = data;
                        targetd(count,(b*5-5)+c) = 1;
                        count = count+1;
                    end
                end
                toc
            end
        end
        countStop(stopCount)= count
        count = 1;
        stopCount = stopCount+1
    end
end
rng(1);
n= size(array,1);
partition = cvpartition(n,'Holdout',n/3);
arrayTrain  = array(training(partition),:);
arrayTest   = array(test(partition),:);
targetTrain = targetd(training(partition),:);
targetTest  = target(test(partition),:);