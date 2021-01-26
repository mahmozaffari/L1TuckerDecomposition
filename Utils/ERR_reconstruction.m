function [ERR] = ERR_reconstruction(X,X_hat)
%ERR_RECONST Summary of this function goes here
%   Detailed explanation goes here
    diff = X - X_hat;
    Xnorm = norm(X(:),'fro')^2;
    ERR = norm(diff(:),'fro')^2 / Xnorm;
end

