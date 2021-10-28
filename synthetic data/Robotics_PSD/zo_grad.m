function g = zo_grad(K, f, mu, m)        
    % Euclidean zeroth-order gradient
    symm = @(X) .5*(X+X');
    d = size(K,1);
    g = zeros(d);
    U = randn(d, d, m);
    for j=1:m
        temp = symm(U(:,:,j));
        g = g + (f(K + temp*mu) - f(K))/mu*temp;
    end
    g = (g + g') / (2*m);
end