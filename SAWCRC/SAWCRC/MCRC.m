function Alpha = MCRC(train_norm,test_norm,sigma)
test_tol=size(test_norm,2);
% sigma=0.2;
lambda=1e-3;
X = train_norm;
Alpha = [];
parfor i=1:test_tol
    y=test_norm(:,i);
    
    dist1=bsxfun(@minus,train_norm,y);
    dist2=sqrt(sum(dist1.^2));
    %     dist2=dist2-max(dist2);
    dist=exp(dist2/sigma);%Ö¸Êý¾àÀë
    %             D=diag(dist.^2);
    D=diag(dist);
    
    P=(X'*X+lambda*D)\X';
    xp=P*y;
    
    Alpha = [Alpha,xp];
end


