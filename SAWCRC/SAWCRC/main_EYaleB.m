clear
clc
close all

addpath(genpath('ompbox10'))
load('..\randomfaces4extendedyaleb.mat');
load('..\Tr_ind_EYaleB.mat')
experiments = size(Tr_ind,1);
acc = zeros(1,experiments);

ClassNum = length(unique(gnd));
sparsity = 50;
sigma = 0.4;

for ii=1:experiments
    ii
    train_ind = logical(Tr_ind(ii,:));
    test_ind = ~train_ind;
    
    training_feats = fea(:,train_ind);
    testing_feats = fea(:,test_ind);
    train_label = gnd(:,train_ind);
    test_label = gnd(:,test_ind);
    
    H_train = full(ind2vec(train_label,ClassNum));
    
    % tic                     % Start computing time from here
    train = normc(training_feats);
    Y = normc(testing_feats);
    
    Phi = train;
    % P = (Phi' * Phi + (lambda* eye(size(Phi,2))))\Phi' ;
    
    % A_check = P * Y;
    
    A_check = MCRC(train,Y,sigma);
    
    G = Phi'*Phi;
    A_hat = omp(Phi'*Y,G,sparsity);
    A_aug = normc(A_check + A_hat);
    Score = H_train * A_aug;
    
    [~,pre_label] = max(Score);
    acc(ii) = sum(pre_label==test_label)/length(test_label)*100
    % toc
end
mean(acc)
std(acc)
