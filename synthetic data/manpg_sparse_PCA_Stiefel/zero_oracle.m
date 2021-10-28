function g_mu = zero_oracle(X, H, mu, m)
    % the zeroth order oracle
    U = randn([size(X), m]);
    g_mu = zeros(size(X));
    for j=1:m
        temp = proj(X, U(:,:,j));
        g_mu = g_mu + (f(retr(X,mu*temp),H)-f(X,H))*temp/mu;
    end
    g_mu = g_mu/m;