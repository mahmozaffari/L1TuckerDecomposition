%% 0-Clear Workspace
clear; clc; close all;

%% 1-Experiment parameters
D = [5,8,5,8,5];        % Tensor dimensionality
d = [3,3,2,2,2];        % low-rank dimensionality
G_std = 3;              % Standard-deviation of the core tensor (Generated by zero-centered Normal distribution)


%% 2-Setup the experiment
% Generates G_true (tensor core), Un_True (true bases)
% creates 'init.mat' file
experiment_folder = './experiment';
I = length(D);
G_true = tensor(normrnd(0, G_std, d));
Un_true = generate_orth_basis(I, D, d);
file_path = fullfile(experiment_folder,'init.mat');
save(file_path,'D','d','G_true','G_std','Un_true');


%% 2-Experiment parameters
sigma_n = 1;            % Noise std
N_o = 10;                % Number of Outlier entries
outlier_sigmas = [0 2 4 5 6 8 10 12 14 16 18 20:4:48];
init_method = 'HOSVD';
itr = 100;
maxit = 100;

for i = 1:length(outlier_sigmas)
    sigma_o = outlier_sigmas(i);
    outdir = fullfile(experiment_folder,['sN' num2str(sigma_n) '-pO' num2str(N_o) '-sO' num2str(sigma_o)]);
    test_Tucker(D , d, Un_true, G_true, outdir, 'R', itr, 'sigma_o', sigma_o, 'sigma_n', sigma_n, 'P_o', N_o, 'P_type', 'count', 'init_method', init_method,'maxit',maxit)
end

%% Test function
function [] = test_Tucker(Ds, ds, varargin)
    DEFAULT_ = -1;

    params = inputParser;
    params.addRequired('Un_true',@(x) iscell(x) & isequal(length(x), length(Ds)))
    params.addRequired('G_true', @(x) isequal(size(x),ds))
    params.addRequired('outdir',@(x) isstr(x))
    params.addParameter('tol',1e-8, @isscalar)
    params.addParameter('maxit',inf,@(x) isscalar(x) & x>0);
    params.addParameter('R',500,@(x) isscalar(x) & x>0);
    params.addParameter('seed',nan,@isscalar);
    params.addParameter('sigma_n',DEFAULT_,@(x) isscalar(x) & x>0);
    params.addParameter('P_o',DEFAULT_,@(x) isscalar(x));
    params.addParameter('P_type','probability',@(x) ismember(x,{'probability','count'}))
    params.addParameter('sigma_o',DEFAULT_,@(x) isscalar(x) & x>=0)
    params.addParameter('init_method','default',@(x) ismember(x,{'default','HOSVD'}))

    params.parse(varargin{:})
    
    %% setup the variables
    % Tensor params
    I = length(Ds);
    G_true = params.Results.G_true;
    Un_true = params.Results.Un_true;
    
    % Noise params
    sigma_n = params.Results.sigma_n;               %% standard deviation of noise
        
    % Outlier params
    sigma_o = params.Results.sigma_o;               %% Standard deviation of outlier
    P_o = params.Results.P_o;                       %% probability of outlier entry/count of outliers
    P_type = params.Results.P_type;                 %% Type of P_o parameter: Fixed count / probability
    
    % outputs
    output_folder = params.Results.outdir;
    if ~exist(output_folder,'dir')
        mkdir(output_folder)
    end
    %% Further Validations:
    assert(isequal(length(Ds),length(ds)))
    if (sigma_o == DEFAULT_) || (sigma_n == DEFAULT_) || (P_o == DEFAULT_)
        error('[ERROR] Parameters sigma-o/sigma-n/P_o are missing.')
    end
    if isequal(P_type, 'probability')
        if P_o < 0 || P_o >= 1
            error('[ERROR] Invalid value is given for P_o. P_o must be between 0 and 1.')
        end
    elseif isequal(P_type, 'count')
        if P_o < 0 || P_o >= prod(Ds)
            error('[ERROR] Invalid value is given for P_o.')
        end
    else
        error('[ERROR] Unknown value is given for P_type argument. P_type must be equal to "probability" or "count"')
    end
    
    %% Pre-test
    onr = getONR_sparse(Ds, 'sigma_o', sigma_o, 'sigma_n', sigma_n, 'P', P_o, 'P_type', P_type);
%     onr_emp = 0;        % empirical ONR
    %% algorithms
    alg_names = {'L1HOOI/L2proj', 'L1HOOI/L2proj(Rand)' ,'L1HOOI/L1proj', 'L1HOOI/L1proj(Rand)','L1HOOI/L1proj(RP)','L2HOOI/L2proj','L2HOOI/L2proj(Rand)','L1HOSVD','L2HOSVD'};
    exclude_for_T1 = {'L1HOSVD','L2HOSVD', 'L2HOOI/L2proj'};
    P = length(alg_names);
    name2id = containers.Map;
    id2name = {};
    for i =1:P
        name2id(alg_names{i}) = i;
        id2name{i,1} = i; id2name{i,2} = alg_names{i};
    end
    
    for j = 1:P
        if ~ismember(alg_names{j}, exclude_for_T1) && isempty(strfind(alg_names{j},'(T=1)'))
            i = i+1;
            name_T1 = [alg_names{j} '(T=1)'];
            name2id(name_T1) = i;
            id2name{i,1} = i; id2name{i,2} = name_T1;
        end
    end
    
    % Algorithm params
    init_method = params.Results.init_method;
    tol = params.Results.tol;
    maxit = params.Results.maxit;
    % randomness params
    seed = params.Results.seed;
    % test params
    itr = params.Results.R;

    % Generate clean tensor
    X_clean = ttm(G_true, Un_true, 1:I);        % Noise free tensor
    % seed
    seed = setSeed('seed',seed);
    
    % Save settings
    save(fullfile(output_folder,'Simulation_settings.mat'),'params','X_clean','G_true','Un_true','X_clean','seed','itr','maxit','tol','init_method','alg_names','Ds','ds','I','P_o','sigma_o','sigma_n','P_type')
    
    %% Result Variables:
    L_perf = zeros(itr,P);
    RTime = zeros(itr,P);
    update_types = cell(itr,P);
    L_metric_hist = cell(itr,P);

    U_hats = cell(itr,P);
    Cores = cell(itr,P);

    ERR_subspc = zeros(itr,P);
    ERR_recons = zeros(itr,P);

    ERR_subspc_hist = cell(itr,P);
    ERR_recons_hist = cell(itr,P);

    for r = 1:itr
        %% Corrupt data
        Z_n = normrnd(0, sigma_n, size(X_clean));           % Additive Noise tensor with mean = 0, and standard deviation = $sigma_n
        X_n = X_clean + Z_n;
        outlier_mask = gen_rand_sparse_indices(P_o, Ds, 'P_type', P_type);
        Z_o = outlier_mask.*normrnd(0, sigma_o, Ds);          % Additive Outlier tensor with mean = 0, and standard deviation = $sigma_o
        %onr_emp = onr_emp + (1/itr)*(norm(Z_o(:),'fro')^2/norm(Z_n(:),'fro')^2);        % Empirical ONR of experiment 
        X_corr = X_n + Z_o;             % X_corr = ttm(G,Un_true,'t') + N + O
        
        % Initialize Uns
        [U0_L1, U0_L2] = initialize_bases(I, Ds, ds, init_method, 'X', X_corr, 'tol', tol);
        
        %%  for Validation (to make sure no over write in idx occurs)
            assert_idx = [];
        %%
        %% L1 HOOI / L2 proj algorithms
        [U, G, ~,~,stats, name, stats_T1] = L1HOOI(X_corr, ds, U0_L1, 'maxit', maxit, 'tol', tol, 'X_clean', X_clean, 'Un_true', Un_true, 'proj', 'L2');
        idx = name2id(name);        assert_idx = [assert_idx idx];
        L_perf(r,idx) = stats.L_metric(end);
        ERR_subspc(r,idx) = stats.SERR(end);
        ERR_recons(r,idx) = stats.RERR(end);
        update_types{r,idx} = stats.update_type;
        L_metric_hist{r,idx} = stats.L_metric;
        ERR_subspc_hist{r,idx} = stats.SERR;
        ERR_recons_hist{r,idx} = stats.RERR;
        U_hats{r,idx} = U;
        Cores{r,idx} = G;
        if ismember([name '(T=1)'],name2id.keys)
            idx2 = name2id([name '(T=1)']);     assert_idx = [assert_idx idx2];
            ERR_subspc(r,idx2) = stats_T1.SERR(end);
            ERR_recons(r,idx2) = stats_T1.RERR(end);
            U_hats{r,idx2} = stats_T1.U;
            L_metric_hist{r,idx2} = stats_T1.L_metric;
            ERR_subspc_hist{r,idx2} = stats_T1.SERR;
            ERR_recons_hist{r,idx2} = stats_T1.RERR;
        end
        
        
        %% L1 HOOI / L2 proj (rand)
        [U, G, ~,~,stats, name,stats_T1] = L1HOOI_rand(X_corr, ds, U0_L1, 'maxit', maxit, 'tol', tol, 'X_clean', X_clean, 'Un_true', Un_true, 'proj', 'L2');
        idx = name2id(name);            assert_idx = [assert_idx idx];
        L_perf(r,idx) = stats.L_metric(end);
        ERR_subspc(r,idx) = stats.SERR(end);
        ERR_recons(r,idx) = stats.RERR(end);
        update_types{r,idx} = stats.update_type;
        L_metric_hist{r,idx} = stats.L_metric;
        ERR_subspc_hist{r,idx} = stats.SERR;
        ERR_recons_hist{r,idx} = stats.RERR;
        U_hats{r,idx} = U;
        Cores{r,idx} = G;
        if ismember([name '(T=1)'],name2id.keys)
            idx2 = name2id([name '(T=1)']);         assert_idx = [assert_idx idx2];
            ERR_subspc(r,idx2) = stats_T1.SERR(end);
            ERR_recons(r,idx2) = stats_T1.RERR(end);
            U_hats{r,idx2} = stats_T1.U;
            L_metric_hist{r,idx2} = stats_T1.L_metric;
            ERR_subspc_hist{r,idx2} = stats_T1.SERR;
            ERR_recons_hist{r,idx2} = stats_T1.RERR;
        end
        
        %% L1 HOOI / L1 proj
        [U, G, ~,~,stats, name,stats_T1] = L1HOOI(X_corr, ds, U0_L1, 'maxit', maxit, 'tol', tol, 'X_clean', X_clean, 'Un_true', Un_true, 'proj','L1');
        idx = name2id(name);            assert_idx = [assert_idx idx];
        L_perf(r,idx) = stats.L_metric(end);
        ERR_subspc(r,idx) = stats.SERR(end);
        ERR_recons(r,idx) = stats.RERR(end);
        update_types{r,idx} = stats.update_type;
        L_metric_hist{r,idx} = stats.L_metric;
        ERR_subspc_hist{r,idx} = stats.SERR;
        ERR_recons_hist{r,idx} = stats.RERR;
        U_hats{r,idx} = U;
        Cores{r,idx} = G;
        if ismember([name '(T=1)'],name2id.keys)
            idx2 = name2id([name '(T=1)']);             assert_idx = [assert_idx idx2];
            ERR_subspc(r,idx2) = stats_T1.SERR(end);
            ERR_recons(r,idx2) = stats_T1.RERR(end);
            U_hats{r,idx2} = stats_T1.U;
            L_metric_hist{r,idx2} = stats_T1.L_metric;
            ERR_subspc_hist{r,idx2} = stats_T1.SERR;
            ERR_recons_hist{r,idx2} = stats_T1.RERR;
        end

        %% L1 HOOI / L1 proj (rand)
        [U, G, ~,~,stats, name,stats_T1] = L1HOOI_rand(X_corr, ds, U0_L1, 'maxit', maxit, 'tol', tol, 'X_clean', X_clean, 'Un_true', Un_true, 'proj','L1');
        idx = name2id(name);                assert_idx = [assert_idx idx];
        L_perf(r,idx) = stats.L_metric(end);
        ERR_subspc(r,idx) = stats.SERR(end);
        ERR_recons(r,idx) = stats.RERR(end);
        update_types{r,idx} = stats.update_type;
        L_metric_hist{r,idx} = stats.L_metric;
        ERR_subspc_hist{r,idx} = stats.SERR;
        ERR_recons_hist{r,idx} = stats.RERR;
        U_hats{r,idx} = U;
        Cores{r,idx} = G;
        if ismember([name '(T=1)'],name2id.keys)
            idx2 = name2id([name '(T=1)']);             assert_idx = [assert_idx idx2];
            ERR_subspc(r,idx2) = stats_T1.SERR(end);
            ERR_recons(r,idx2) = stats_T1.RERR(end);
            U_hats{r,idx2} = stats_T1.U;
            L_metric_hist{r,idx2} = stats_T1.L_metric;
            ERR_subspc_hist{r,idx2} = stats_T1.SERR;
            ERR_recons_hist{r,idx2} = stats_T1.RERR;
        end
        
        %% L1 HOOI / L1 proj (RP)
        [U, G, ~,~,stats, name,stats_T1] = L1HOOI(X_corr, ds, U0_L1, 'maxit', maxit, 'tol', tol, 'X_clean', X_clean, 'Un_true', Un_true, 'proj','L1','selection','random');
        idx = name2id(name);                assert_idx = [assert_idx idx];
        L_perf(r,idx) = stats.L_metric(end);
        ERR_subspc(r,idx) = stats.SERR(end);
        ERR_recons(r,idx) = stats.RERR(end);
        update_types{r,idx} = stats.update_type;
        L_metric_hist{r,idx} = stats.L_metric;
        ERR_subspc_hist{r,idx} = stats.SERR;
        ERR_recons_hist{r,idx} = stats.RERR;
        U_hats{r,idx} = U;
        Cores{r,idx} = G;
        if ismember([name '(T=1)'],name2id.keys)
            idx2 = name2id([name '(T=1)']);             assert_idx = [assert_idx idx2];
            ERR_subspc(r,idx2) = stats_T1.SERR(end);
            ERR_recons(r,idx2) = stats_T1.RERR(end);
            U_hats{r,idx2} = stats_T1.U;
            L_metric_hist{r,idx2} = stats_T1.L_metric;
            ERR_subspc_hist{r,idx2} = stats_T1.SERR;
            ERR_recons_hist{r,idx2} = stats_T1.RERR;
        end
        
        %% L2 HOOI / L2 proj algorithms
        T = tucker_als(X_corr, ds, 'init', U0_L2, 'maxiters',maxit, 'tol', tol);
        U = T.U;
        G = T.core;
        Xhat = ttm(G, U);
        name = 'L2HOOI/L2proj';
        idx = name2id(name);                assert_idx = [assert_idx idx];
        L_perf(r,idx) = norm(G)^2;
        ERR_subspc(r,idx) = ERR_subspace(Un_true, U, ds);
        ERR_recons(r,idx) = ERR_reconstruction(X_clean, Xhat);
        update_types{r,idx} = stats.update_type;
        L_metric_hist{r,idx} = [];
        ERR_subspc_hist{r,idx} = [];
        ERR_recons_hist{r,idx} = [];
        U_hats{r,idx} = U;
        Cores{r,idx} = G;
        

        %% L1HOSVD
        name = 'L1HOSVD';
        idx = name2id(name);                assert_idx = [assert_idx idx];
        if isequal( init_method, 'HOSVD' )
            U = U0_L1;
        else
            T = hosvd(X_corr, tol, 'rank', ds, 'verbosity',0);
            U = L1HOSVD(X_corr, ds, T.U, 'maxit', 1000, 'tol', tol);          % initialized by HOSVD
        end
        G = ttm(X_corr, U,'t');
        Xhat = ttm(G, U);
        U_hats{r,idx} = U;
        Cores{r,idx} = G;
        ERR_recons(r,idx) = ERR_reconstruction(X_clean, Xhat);
        ERR_subspc(r,idx) = ERR_subspace(Un_true, U, ds);
        
        %% HOSVD
        name = 'L2HOSVD';
        idx = name2id(name);                    assert_idx = [assert_idx idx];
        if isequal( init_method, 'HOSVD' )
            U = U0_L2;
        else
            T = hosvd(X_corr, tol, 'rank', ds, 'verbosity',0);
            U = T.U;
        end
        
        G = ttm(X_corr, U,'t');
        Xhat = ttm(G, U);
        U_hats{r,idx} = U;
        Cores{r,idx} = G;
        ERR_recons(r,idx) = ERR_reconstruction(X_clean,Xhat);
        ERR_subspc(r,idx) = ERR_subspace(Un_true,U,ds);
        
        
        %% Validation
        if ~isequal(length(assert_idx) ,length(unique(assert_idx)))
            error('[ERROR]')
        end
        %%
    end
    names = name2id.keys;
    P = length(names);
    save(fullfile(output_folder,'Results.mat'),'names','id2name','ERR_recons_hist','ERR_subspc_hist','onr','ERR_subspc','ERR_recons','Cores','U_hats','L_perf','L_metric_hist','alg_names','name2id','update_types','RTime','P')

end