function r = retr(X,Y)
    % retraction to Stiefel manifold
    [n, p] = size(X);
    [U,~,V] = svd(X+Y);
    r = U*eye(n,p)*V.';
    
    % This is to grassmann (also to Stiefel)
%     [~, p] = size(X);
%     [Q, ~] = qr(X+Y);
%     r = Q(:,1:p);
end