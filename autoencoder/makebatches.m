% Version 1.000
%
% Code provided by Ruslan Salakhutdinov and Geoff Hinton
%
% Permission is granted for anyone to copy, use, modify, or distribute this
% program and accompanying programs and documents for any purpose, provided
% this copyright notice is retained and prominently displayed, along with
% a note saying that the original programs are available from our
% web page.
% The programs and documents are distributed without any warranty, express or
% implied.  As the programs were written for research purposes only, they have
% not been tested to the degree that would be advisable in any important
% application.  All use of these programs is entirely at the user's own risk.

digitdata=[]; 
load digit0; digitdata = [digitdata; D]; % 5923*784 % 要行优先di 向下堆叠
load digit1; digitdata = [digitdata; D]; % 12665*784  才能还原28*28的图像
load digit2; digitdata = [digitdata; D]; % 18623*784
load digit3; digitdata = [digitdata; D]; % 24754*784
load digit4; digitdata = [digitdata; D]; % 30596*784
load digit5; digitdata = [digitdata; D]; % 36017*784
load digit6; digitdata = [digitdata; D]; % 41935*784
load digit7; digitdata = [digitdata; D]; % 48200*784
load digit8; digitdata = [digitdata; D]; % 54051*784
load digit9; digitdata = [digitdata; D]; % 60000*784
digitdata = digitdata/255;

totnum=10000;%size(digitdata,1);
fprintf(1, 'Size of the training dataset= %5d \n', totnum);

rand('state',0); %so we know the permutation of the training data
randomorder=randperm(size(digitdata,1), totnum);

% batchsize = 100;
numbatches=totnum/batchsize; 
numdims  =  size(digitdata,2);
batchdata = zeros(batchsize, numdims, numbatches);
for b=1:numbatches
  batchdata(:,:,b) = digitdata(randomorder(1+(b-1)*batchsize:b*batchsize), :);
end;
clear digitdata;

digitdata=[];
load test0; digitdata = [digitdata; D];
load test1; digitdata = [digitdata; D];
load test2; digitdata = [digitdata; D];
load test3; digitdata = [digitdata; D];
load test4; digitdata = [digitdata; D];
load test5; digitdata = [digitdata; D];
load test6; digitdata = [digitdata; D];
load test7; digitdata = [digitdata; D];
load test8; digitdata = [digitdata; D];
load test9; digitdata = [digitdata; D];
digitdata = digitdata/255;

totnum=size(digitdata,1);
fprintf(1, 'Size of the test dataset= %5d \n', totnum);

rand('state',0); %so we know the permutation of the training data
randomorder=randperm(totnum);

numbatches=totnum/batchsize;
numdims  =  size(digitdata,2);
% batchsize = 100;
testbatchdata = zeros(batchsize, numdims, numbatches);
for b=1:numbatches
  testbatchdata(:,:,b) = digitdata(randomorder(1+(b-1)*batchsize:b*batchsize), :);
end;
clear digitdata;


%%% Reset random seeds 
rand('state',sum(100*clock)); 
randn('state',sum(100*clock)); 



