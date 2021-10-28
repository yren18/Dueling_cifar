function P = proj(~, X)
    % projection of X to tangent space at U
    symm = @(X) .5*(X+X');
    P = symm(X);