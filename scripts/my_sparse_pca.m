function [Y, V] = my_sparse_pca(X, k, step_size)

%X = X - repmat(mean(X,1),size(X,1),1);

% [~, nn] = size(X);
% 
% parfor j = 1 : nn
%     j
%     X(:, j) = X(:, j) - mean(X(:, j));
% end
tic;
%X = bsxfun(@minus, X, mean(X));
[~, n] = size(X);
up_bound = floor(n / step_size) * step_size;
for i = 1 : step_size : up_bound
    X(:, i : i + step_size - 1) = bsxfun(@minus, X(:, i : i + step_size - 1), mean(X(:, i : i + step_size - 1)));
end
i = i + step_size;
if up_bound < n
    X(:, i : n) = bsxfun(@minus, X(:, i : n), mean(X(:, i : n)));
end
toc

tic
C = X'*X./(size(X,1)-1); 
toc
tic;
opts.tol = 1e-4;
[F, V] = eigs(C, k, 'la', opts);
toc
Y = X*F;

end