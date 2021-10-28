function f = f(X,H)
    f = -0.5*trace(X.'*H*X);
end