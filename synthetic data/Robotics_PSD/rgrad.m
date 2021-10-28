function g = rgrad(X,A)
    
    N = size(A, 3);
    logsum = zeros(size(X,1));

    for i = 1 : N
        logsum = logsum + logm(A(:, :, i) \ X);
    end

    g = X*logsum;
    g = (g+g')/2;
    g = g / N;
end