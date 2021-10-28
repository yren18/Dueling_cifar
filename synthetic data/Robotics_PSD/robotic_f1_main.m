%% Test on zeroth-order Riemannian problem of robotics, f1
%  Modified from http://proceedings.mlr.press/v100/jaquier20a/jaquier20a.pdf
clc; close all; clear

%% Generate data and initialization
% Data
d = 2;
if d==2
    hat_p = [0.66, -0.01].';
    dot_p = [0, 0].'; fe = [0, 20].';
else
    hat_p = [0.66, -0.01, 0.69].';
    dot_p = [0, 0, 0].'; fe = [0, 20, -20].';
end
wp = 1; wd = 1e-13; wc = 1e-4;
p = @(K) hat_p - dot_p - K \ fe;
f = @(K) wp*norm(hat_p - p(K))^2 + wd*det(K) + wc*cond(K);

% initialization
max_iter = 200;
eta1 = 3e-4; eta = 1e-3; % stepsize
m = 10; mu = 1e-7; % zo parameters
val_list = zeros(100, max_iter, 2);
norm_list = zeros(100, max_iter, 2);

%% Optimization steps
for rep = 1:100
    K = eye(d); % initialize as identity matrix
    K1 = K;
    for k = 1:max_iter
    %% Zeroth-order RGD
    G = zo_grad(K1, f, mu, m);
    K1 = K1-eta1*G;
    [V, D] = eig(K1);
    K1 = V*min(max(D, 0.001), 5)*V.';
    
    %% Zeroth-order RGD
    G = rzo_grad(K, f, mu, m);
    K = retr(K, -eta*G);
    [V, D] = eig(K);
    K = V*min(max(D, 0.001), 5)*V.';
    
    %% print infomation for current iteration
    val_list(rep, k, 1) = f(K1); norm_list(rep, k, 1) = norm(K1);
    val_list(rep, k, 2) = f(K); norm_list(rep, k, 2) = norm(K);
%     if mod(k, 10) == 0
%         fprintf("Iteration number %d, function value: %f \n", k, val_list(k, 2));
%     end
%     G1 = rgrad(U1, A); G2 = rgrad(U2, A); G3 = rgrad(U3, A);
%     norm1(k) = norm(G1,'fro'); norm2(k) = norm(G2,'fro'); norm3(k) = norm(G3,'fro'); 
%     cost1(k) = cost(U1,N,A); cost2(k) = cost(U2,N,A); cost3(k) = cost(U3,N,A);
%     disp(join(["Iteration", num2str(k), "; norm for first order:", num2str(norm1(k)),...
%         "; norm for zeroth order:", num2str(norm2(k)), "Full gradient:", num2str(norm3(k))]));
%     disp(join(["func.val. for first order:", num2str(cost1(k)),...
%         "; func.val. for zeroth order:", num2str(cost2(k)), "Function val. for full:", num2str(cost3(k))]));
    end
end


% % save the result
% filename = [num2str(n),'_',num2str(p),'_result_norm.mat'];
% save(filename,'norm1','norm2','norm3');

%% Plots
% figure;
% semilogy(1:(max_iter+1), [f(eye(d)); val_list(:,1)]);
% hold on
% semilogy(1:(max_iter+1), [f(eye(d)); val_list(:,2)]);
% title("Function value");
% legend("ZO-GD","ZO-RGD");
mean_val_list = mean(val_list, 1);
figure;
semilogy(1:(max_iter+1), [f(eye(d)), mean_val_list(1, :, 1)], 'b');
hold on
semilogy(1:(max_iter+1), [f(eye(d)), mean_val_list(1, :, 2)]), 'r';
hold on
legend("ZO-GD", "ZO-RGD")

inBetween = [f(eye(d)), quantile(val_list(:, :, 1),0.1,1), fliplr(quantile(val_list(:, :, 1),0.9,1)), f(eye(d))];
% Choose a number between 0 (invisible) and 1 (opaque) for facealpha.  
h1 = fill([1:(max_iter+1), fliplr(1:(max_iter+1))], inBetween, 'b');
set(h1,'facealpha',.5)
hold on
inBetween = [f(eye(d)), quantile(val_list(:, :, 2),0.1,1), fliplr(quantile(val_list(:, :, 2),0.9,1)), f(eye(d))];
h2 = fill([1:(max_iter+1), fliplr(1:(max_iter+1))], inBetween, 'r');
set(h2,'facealpha',.5)
hold on
set(get(get(h1,'Annotation'),'LegendInformation'),'IconDisplayStyle','off');
set(get(get(h2,'Annotation'),'LegendInformation'),'IconDisplayStyle','off');
title("function value curve, dimension d=" + num2str(d))