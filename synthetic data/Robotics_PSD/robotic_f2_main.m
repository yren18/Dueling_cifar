%% Test on zeroth-order Riemannian problem of robotics, f1
%  Modified from http://proceedings.mlr.press/v100/jaquier20a/jaquier20a.pdf
clc; close all; clear

%% Generate data and initialization
% Data
d = 3;
hat_p = [0.66, -0.01, 0.69].';
dot_p = [0, 0, 0].';
wp = 1; wd = 1e-13; wc = 1e-4; fe = [0, 20, -20].';
p = @(K) hat_p - dot_p - K \ fe;
f = @(K) wp*norm(hat_p - p(K))^2;

% initialization
max_iter = 200;
K = eye(d); % initialize as identity matrix
eta = 1e-3; % stepsize
m = 10; mu = 1e-8; % zo parameters
val_list = zeros(max_iter, 1);
norm_list = zeros(max_iter, 1);

%% Optimization steps
for k = 1:max_iter
    %% Zeroth-order SGD
    G = -rzo_grad(K, f, mu, m);
    K = retr(K, eta*G);
    [V, D] = eig(K);
    K = V*min(max(D, 0.001), 5)*V.';
    
    %% print infomation for current iteration
    val_list(k) = f(K); norm_list(k) = norm(K);
    if mod(k, 10) == 0
        fprintf("Iteration number %d, function value: %f \n", k, val_list(k));
    end
%     G1 = rgrad(U1, A); G2 = rgrad(U2, A); G3 = rgrad(U3, A);
%     norm1(k) = norm(G1,'fro'); norm2(k) = norm(G2,'fro'); norm3(k) = norm(G3,'fro'); 
%     cost1(k) = cost(U1,N,A); cost2(k) = cost(U2,N,A); cost3(k) = cost(U3,N,A);
%     disp(join(["Iteration", num2str(k), "; norm for first order:", num2str(norm1(k)),...
%         "; norm for zeroth order:", num2str(norm2(k)), "Full gradient:", num2str(norm3(k))]));
%     disp(join(["func.val. for first order:", num2str(cost1(k)),...
%         "; func.val. for zeroth order:", num2str(cost2(k)), "Function val. for full:", num2str(cost3(k))]));
end

% % save the result
% filename = [num2str(n),'_',num2str(p),'_result_norm.mat'];
% save(filename,'norm1','norm2','norm3');

%% Plots
figure;
semilogy(1:(max_iter+1), [f(eye(d)); val_list]);
title("Function value");