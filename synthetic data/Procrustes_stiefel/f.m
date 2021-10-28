function f = f(X,A,B)
    f = (norm(A*X-B,'fro'))^2;
end