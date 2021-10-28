function g_mu = zero_oracle(X, B, mu, m)
    % the zeroth order oracle
    f = @(X) -sum(sum(X.*(B*X)));
    proj = @(X,Y) Y - X*(X.'*Y+Y.'*X)/2;
    U = randn([size(X), m]);
    g_mu = zeros(size(X));
    for j=1:m
        temp = proj(X, U(:,:,j));
        g_mu = g_mu + (f(retr(X,mu*temp))-f(X))*temp/mu;
    end
