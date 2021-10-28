function nabla = nablaf(X,A,B)
    nabla = 2*A.'*(A*X-B);
end