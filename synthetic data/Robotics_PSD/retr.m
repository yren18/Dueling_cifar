function R = retr(U, X)
    % retraction of X at U
    symm = @(X) .5*(X+X');
    R = symm(U + X + 0.5 * X * (U \ X));