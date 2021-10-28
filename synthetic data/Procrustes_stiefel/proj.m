function proj = proj(X,Y,n)
% Projecting Y onto the tangent space of Stiefel manifold, at X
    proj = X*(X.'*Y-Y.'*X)/2+(eye(n)-X*X.')*Y;
end