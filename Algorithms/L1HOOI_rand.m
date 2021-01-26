function [U,G,Xhat,L1met_end,stats,funcname,stats_T1] = L1HOOI_rand(X, Ks, Uin, varargin)
    
    params = inputParser;
    params.addParameter('tol',1e-8,@isscalar);
    params.addParameter('maxit',1000,@(x) isscalar(x) & x>0);
    params.addParameter('X_clean',[], @(x) isequal(size(x),size(X)))
    params.addParameter('Un_true',{},@(x) iscell(x) & isequal(length(x),length(Ks)))
    params.addParameter('T',inf, @(x) isscalar(x) & x>0);
    params.addParameter('proj','L2', @(x) ismember(x,{'L1','L2'}))
    
    params.parse(varargin{:})
    maxit = params.Results.maxit;
    tol = params.Results.tol;
    X_clean = params.Results.X_clean;
    Un_true = params.Results.Un_true;
    T = params.Results.T;
    proj = params.Results.proj;
    
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
    %%
    
    G = ttm(X,Uin,'t');    l1met0 = sum(abs(G.data(:)));    serr0 = ERR_subspace(Un_true, Uin, Ks);
    rerr0 = ERR_reconstruction(X_clean,ttm(G,Uin).data);
    
    L1met_o = zeros(1,maxit); L1met_i = zeros(1,maxit);
    pmet_o = 0;
    update_types = zeros(3,maxit);
    SERR = zeros(1,maxit);
    RERR = zeros(1,maxit);
    
    Ds = size(X);
    n = ndims(X);
    U = Uin;
    t3 = 0;  tt = 0;     ttt = 0;
    % t3: increment per each outer while loop iteration [increment 1 per each (U1B)(U2B)...(UnB)]
    % tt: increment per each (UiB) block iterations
    % ttt: increment per each individual U update.
    i = 0;
    % tt is the number of (UB) block updates
    func_start = tic;
    T1flag = false;
    
    while t3<maxit;   t3 = t3 + 1;
        previous_basis_i = i;
        while previous_basis_i == i
            i = randi(n);   
        end
        
        tt = tt + 1;
        
        K = Ks(i);
        P = U;      P{i} = eye(Ds(i),Ds(i));
        % Ui updates
        if K == Ds(i)
            Ui = eye(Ds(i));
        else
            %% L1 projection
            [Si] = transform_X3(X, U, Ds, Ks, i, proj);
            Si = Si.data;
            %% Begin L1-PCA
            [Ui, l1met, updates, ~,serr,rerr] = L1PCA(X, Si, Ks, U, i, maxit,tol,Un_true, X_clean);   t_ = length(l1met);
            L1met_i(ttt+1:ttt+t_) = l1met;
            SERR(ttt+1:ttt+t_) = serr;
            RERR(ttt+1:ttt+t_) = rerr;
            update_types(1,2*ttt+1:2*ttt+2*t_) = updates;   update_types(2,2*ttt+1:2*ttt+2*t_) = tt; update_types(3,2*ttt+1:2*ttt+2*t_) = t3;
            ttt = ttt + t_;
            % End L1-PCA
        end
        U{i} = Ui;

        G = ttm(X,U,'t');
        L1met_o(tt)=  sum(abs(G(:)));   % record L1metric at the end of each L1PCA run
%             SERR(tt) = ERR_subspace(Un_true, U, Ks);
%         RERR(tt) = ERR_reconstruction(X_clean, ttm(G,U).data);
        if ~T1flag
            T1flag = length(unique(update_types(1,1:2*ttt))) == (n+1);      % +1 because one of the elements is always 0
        end
        if T1flag==1
            stats_T1.RERR = [rerr0 RERR(1:ttt)];
            stats_T1.SERR = [serr0 SERR(1:ttt)];
            stats_T1.L_metric = [l1met0  L1met_i(1:ttt)];
            stats_T1.U = U;
            T1flag = 2;
        end
        % Convergence check for outer loop
        if ((abs(L1met_o(tt) - pmet_o)/abs(pmet_o)) < tol) || (t3>=T)
            break;
        else
            pmet_o = L1met_o(tt);
        end
        
    end    % End outer while
    
    Xhat = ttm(G,U).data;
    G = G.data;
    L1met_i = [l1met0 L1met_i(1:ttt)];
    L1met_o = [l1met0 L1met_o(1:tt)];
    L1met_end = L1met_o(end);
    update_types = update_types(:,1:2*ttt);
    SERR = [serr0 SERR(1:ttt)];
    RERR = [rerr0 RERR(1:ttt)];

    stats = struct();
    stats.exec_time = toc(func_start); 
    stats.update_type = update_types;
    stats.L_metric = L1met_i;
    stats.SERR = SERR;
    stats.RERR = RERR;
    stats.U0ERR = [];
    stats.change_in_U = [];
    stats.change_in_U_2 = [];
    if ~exist('stats_T1','var')
        stats_T1.RERR = RERR;
        stats_T1.SERR = SERR;
        stats_T1.L_metric = L1met_i;
        stats_T1.U = U;
    end
    %%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%% length(update_types) = 2*(length(L1met_i) - 1)            %%%
    %%% t3 = length(L1met_o) - 1                                  %%%
    %%% L1met_o records L1metric at the end of each (U_i,B) block %%%
    %%% L1met_oo records L1metric at the end of each outer loop   %%%
    %%% i.e. (once all (U_i,B) blocks are executed)               %%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end

function [G] = transform_X3(X,U, D, d,mode, projection)
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

function [U,S,V]=mSVD(X,K)
       [UK,SK,VK]=svd(X,'econ');
       S=SK(1:K,1:K);
       U=UK(:,1:K);
       V=VK(:,1:K);
end

function [Q, L1met, update_types, l1pcamet,serr,rerr]=L1PCA(X,G, Ks, U, j, maxit,tol,Un_true,X_clean)
    P = U;
    Q = P{j};
    K = Ks(j);
    l1pcamet = zeros(1,maxit);   pmet = 0; % met in original L1PCA code
    L1met = zeros(1,maxit); % met after update U
    update_types = zeros(1,2*maxit);
    serr = zeros(1,maxit);      rerr = zeros(1,maxit);
    t = 0;
    while t<maxit
        t = t + 1;
        b_ = G'*Q;
        B = sign(b_);     update_types(2*(t-1)+1) = 0;
        l1pcamet(t)=sum(abs(b_(:)));
        [UK,~,VK]=mSVD(G*B,K);
        Q=UK*VK';
        P{j} = Q;       update_types(2*t) = j;
        core = ttm(X,P,'t');
        L1met(t) = sum(abs(core.data(:)));
        serr(t) = ERR_subspace(Un_true, P, Ks);
        rerr(t) = ERR_reconstruction(X_clean, ttm(core,P).data);
        if abs(l1pcamet(t)-pmet)/abs(pmet) <= tol
            break;
        else
            pmet = l1pcamet(t);
        end
    end
    l1pcamet = l1pcamet(1:t);
    L1met = L1met(1:t);
    update_types = update_types(1:2*t);
    serr = serr(1:t);
    rerr = rerr(1:t);
end