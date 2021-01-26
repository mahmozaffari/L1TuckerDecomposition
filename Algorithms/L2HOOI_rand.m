function [U,core, Xhat, metrics, stats, funcname,stats_T1] = L2HOOI_rand(X, R, Uin, varargin)
%TUCKER_ALS Higher-order orthogonal iteration.
%
%   T = TUCKER_ALS(X,R) computes the best rank-(R1,R2,..,Rn)
%   approximation of tensor X, according to the specified dimensions
%   in vector R.  The input X can be a tensor, sptensor, ktensor, or
%   ttensor.  The result returned in T is a ttensor.
%
%   T = TUCKER_ALS(X,R,Uin,'param',value,...) specifies optional parameters and
%   values. Valid parameters and their default values are:
%      'tol' - Tolerance on difference in fit {1.0e-4}
%      'maxiters' - Maximum number of iterations {50}
%      'dimorder' - Order to loop through dimensions {1:ndims(A)}
%
%   [T,U0] = TUCKER_ALS(...) also returns the initial guess.
%
%   Examples:
%   X = sptenrand([5 4 3], 10);
%   T = tucker_als(X,2);        %<-- best rank(2,2,2) approximation 
%   T = tucker_als(X,[2 2 1]);  %<-- best rank(2,2,1) approximation 
%   T = tucker_als(X,2,'dimorder',[3 2 1]);
%   T = tucker_als(X,2,'dimorder',[3 2 1],'init','nvecs');
%   U0 = {rand(5,2),rand(4,2),[]}; %<-- Initial guess for factors of T
%   T = tucker_als(X,2,'dimorder',[3 2 1],'init',U0);
%
%   <a href="matlab:web(strcat('file://',...
%   fullfile(getfield(what('tensor_toolbox'),'path'),'doc','html',...
%   'tucker_als_doc.html')))">Documentation page for Tucker-ALS</a>
%
%   See also HOSVD, TTENSOR, TENSOR, SPTENSOR, KTENSOR.
%
%MATLAB Tensor Toolbox. Copyright 2018, Sandia Corporation.


% Extract number of dimensions and norm of X.
N = ndims(X);
normX = norm(X);
D = size(X);

%% Set algorithm parameters from input or by using defaults
params = inputParser;
params.addParameter('tol',1e-4,@isscalar);
params.addParameter('maxit',50,@(x) isscalar(x) & x > 0);
params.addParameter('dimorder',1:N,@(x) isequal(sort(x),1:N));
params.addParameter('X_clean',[], @(x) isequal(size(x),size(X)))
params.addParameter('Un_true',{},@(x) iscell(x) & isequal(length(x),length(R)))
params.addParameter('proj','L2',@(x) ismember(x,{'L1','L2'}))
params.addParameter('T',inf,@(x) isscalar(x) & x>0)
params.parse(varargin{:});

%%
    
if ~isempty(params.Results.Un_true)
    se_flag = true;
    Un_true = params.Results.Un_true;
    serr0 = ERR_subspace(Un_true, Uin, R);
else
    serr0 = [];
    se_flag = false;
end
if ~isempty(params.Results.X_clean)
    re_flag = true;
    X_clean = params.Results.X_clean;
    rerr0 = ERR_reconstruction(X_clean,ttm(ttm(X,Uin,'t'),Uin).data);
else
    rerr0 = [];
    re_flag = false;
end




%% Copy from params object
fitchangetol = params.Results.tol;
maxiters = params.Results.maxit;
dimorder = params.Results.dimorder;
T = params.Results.T;
proj = params.Results.proj;

if numel(R) == 1
    R = R * ones(N,1);
end

%% set function name
st = dbstack;
funcname = st.name;
funcname = [funcname(1:strfind(funcname,'_')-1) '/' proj 'proj(Rand)' ];
if T == 1
    funcname = [funcname '(T=1)'];
end
%% adjust T for L1proj
if isequal(proj,'L1')
    T = min(T,10);
end
%% Error checking 
% Error checking on maxiters
if maxiters < 0
    error('OPTS.maxiters must be positive');
end

% Error checking on dimorder
if ~isequal(1:N,sort(dimorder))
    error('OPTS.dimorder must include all elements from 1 to ndims(X)');
end

%% Set up and error checking on initial guess for U.        %% [Mahsa]
if numel(Uin) ~= N
    error('U.init does not have %d cells',N);
end
for n = dimorder(1:end)
    if ~isequal(size(Uin{n}),[size(X,n) R(n)])
        error('U.init{%d} is the wrong size',n);
    end
end
%% Set up for iterations - initializing U and the fit.
U = Uin;                                            %% [Mahsa]
fit = 0;

core = ttm(X, U, 't');                              %% [Mahsa]

met0 = sqrt( normX^2 - norm(core)^2 );              %% [Mahsa]
metrics = zeros(1,maxiters);                        %% [Mahsa]
update_type = zeros(1,maxiters);                    %% [Mahsa]
Subspace_errors = zeros(1,maxiters);                %% [Mahsa]
Reconstruction_errors = zeros(1,maxiters);          %% [Mahsa]

%% Main Loop: Iterate until convergence
iter = 1;
func_start = tic;
tt=0;
n = 0;
T1flag = false;
while iter <= min(maxiters, T)

    fitold = fit;

    % Iterate over all N modes of the tensor
    
        previous_basis_i = n;
        while previous_basis_i == n
            n = randi(N);   
        end
        
        tt = tt + 1;
        
        Utilde = transform_X3(X, U, D, R, n, proj); Utilde = tensor(Utilde);    %% [Mahsa]  %         old:Utilde = ttm(X, U, -n, 't');
        
        % Maximize norm(Utilde x_n W') wrt W and
        % keeping orthonormality of W
        U{n} = nvecs(Utilde,n,R(n));
        update_type(tt) = n;
        if se_flag
            Subspace_errors(tt) = ERR_subspace(Un_true, U, R);
        end
        if re_flag
            Xhat = ttm(ttm(X,U,'t'),U);
            Reconstruction_errors(tt) = ERR_reconstruction(X_clean,Xhat);
        end
        metrics(tt) = sqrt( normX^2 - norm(ttm(X, U, 't'))^2 );            %% [Mahsa]

    % Assemble the current approximation
    core = ttm(X, U, 't');          %% [Mahsa]

    % Compute fit
    normresidual = sqrt( normX^2 - norm(core)^2 );
    fit = 1 - (normresidual / normX); %fraction explained by model
    fitchange = abs(fitold - fit);
    
    if ~T1flag
        T1flag = length(unique(update_type(1:tt))) == n;
    end
    if T1flag==1
        stats_T1.RERR = [rerr0 Reconstruction_errors(1:tt)];
        stats_T1.SERR = [serr0 Subspace_errors(1:tt)];
        stats_T1.L_metric = [met0 metrics(1:tt)];
        stats_T1.U = U;
        T1flag = 2;
    end

    % Check for convergence
    if (iter > 1) && (fitchange < fitchangetol)
        break;
    end
    iter = iter + 1;
end

% T = ttensor(core, U);
Xhat = ttm(core, U);
stats = struct();
stats.exec_time = toc(func_start); 
stats.update_type = update_type(1:tt);
stats.L_metric = [met0 metrics(1:tt)];
stats.SERR = [serr0 Subspace_errors(1:tt)];
stats.RERR = [rerr0 Reconstruction_errors(1:tt)];
stats.U0ERR = [];
stats.change_in_U = [];
stats.change_in_U_2 = [];
if ~exist('stats_T1','var')
    stats_T1.RERR = stats.RERR;
    stats_T1.SERR = stats.SERR;
    stats_T1.L_metric = stats.L_metric;
    stats_T1.U = U;
end
end

function [G] = transform_X3(X,U, D, d, mode, projection)
    I = length(U);
    modes = [1:mode-1 mode+1:I];

    if ~exist('projection','var')
        projection = 'L2';
    end
    
    if isequal(projection,'L2')
        G = ttm(tensor(X),U(modes),modes,'t');
        G = tenmat(G,mode);
    elseif isequal(projection,'L1')
        U_tilde = 1;
        for i = modes(end:-1:1)
            U_tilde = kron(U_tilde,U{i});
        end
        X = tenmat(X,mode).data;    X = X';
        d_ = d; d_(mode) = D(mode);
        G_mode = tenmat(tensor(zeros(d_)),mode); assert(size(G_mode,1) == size(X,2)); assert(size(G_mode,2) == size(U_tilde,2));
        
        M = size(U_tilde,1);
        m = size(U_tilde,2);

        parfor j = 1:size(X,2)
            x = X(:,j);
            f = [zeros(1,m) ones(1,M)];   % [x,t] -> sum(t)
            A = [U_tilde -1*eye(M) ; -U_tilde -1*eye(M)];
            b = [x;-x];
            gt = linprog(f,A,b);
            G_mode(j,:) = gt(1:m);
        end
        
%         G = G_mode;
        G_mode = tensor(G_mode);
        G = ttm(G_mode,U([1:mode-1, mode+1:I]),[1:mode-1, mode+1:I]);
        G = tenmat(G,mode);
        
    else
        error(['Unknown projection type: ' projection])
    end
end
