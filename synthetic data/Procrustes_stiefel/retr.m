function r = retr(X,Y,n,p)
% retraction to Stiefel manifold
    [U,~,V] = svd(X+Y);
    r = U*eye(n,p)*V.';
end