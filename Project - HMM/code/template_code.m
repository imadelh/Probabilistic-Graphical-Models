generate_barcode;

S = length(patterns);

patternLengths = zeros(S,1);
for s = 1:S
    patternLengths(s) = length(patterns{s});
end

M = 6;
C = max(patternLengths);
NumStates = C*S*M;

%Enumerate all the states
States = zeros(NumStates,3) -1 ;
StatesInv = zeros(C,S,M) -1;

ix = 1;
for c = 1:C
    for s = 1:S
        for m = 1:M
            if( (s > 5) || (s <=5 && m ==1) )
                if( c<=patternLengths(s))
                    States(ix,:) = [c s m];
                    StatesInv(c,s,m) = ix;
                    ix = ix+1;
                end
            end
        end
    end
end

NumStates = ix -1;
States = States(1:NumStates,:);

%warning: NumStates will be less than S*M*C, because not all possible
%[s,m,c] triples are valid. 


%% Part1: Fill the transition matrix A

%mapping states to binary numbers, which will be useful for computing the
%likelihood
f_kst = zeros(NumStates,1); 


A = zeros(NumStates);

for i = 1:NumStates
    
    c = States(i,1);
    s = States(i,2);
    m = States(i,3);
    
    patternLen = patternLengths(s);
    f_kst(i) = patterns{s}(c); %determines if this state is black or white
    
    %example:
    if(s == 1) %starting quiet zone
        
        if(c == patternLen)
            
            for ss = [1 3] %the next states can only be either starting quiet zone, or the starting guard
                s_next = ss;
                c_next = 1;
                m_next = 1;
                
                nextStateIx = StatesInv(c_next,s_next,m_next);
                A(nextStateIx,i) = (1/2);
            end
            
        else
            c_next = c+1;
            s_next = s;
            m_next = m;
            
            nextStateIx = StatesInv(c_next,s_next,m_next);
            A(nextStateIx,i) = 1;
        end
        
        
    elseif(s == 2) %ending quiet zone
        
        if(c == patternLen)
            s_next = 2;
            c_next = 1;
            m_next = 1;

            nextStateIx = StatesInv(c_next,s_next,m_next);
            A(nextStateIx,i) = 1;
                        
        else
            c_next = c+1;
            s_next = s;
            m_next = m;
            
            nextStateIx = StatesInv(c_next,s_next,m_next);
            A(nextStateIx,i) = 1;
        end
        
        
    elseif(s== 3) %starting guard
        
        if(c == patternLen)
            
            for ss = [6:15] %the next states can only be either starting quiet zone, or the starting guard
                s_next = ss;
                c_next = 1;
                m_next = 1; %% Modified from 1 to m+1
                
                nextStateIx = StatesInv(c_next,s_next,m_next);
                A(nextStateIx,i) = (1/10);
            end
            
        else
            c_next = c+1;
            s_next = s;
            m_next = m;
            
            nextStateIx = StatesInv(c_next,s_next,m_next);
            A(nextStateIx,i) = 1;
        end
        
    elseif(s== 4) %ending guard
        
        if(c == patternLen)
            s_next = 2;
            c_next = 1;
            m_next = 1;

            nextStateIx = StatesInv(c_next,s_next,m_next);
            A(nextStateIx,i) = 1;
                        
        else
            c_next = c+1;
            s_next = s;
            m_next = m;
            
            nextStateIx = StatesInv(c_next,s_next,m_next);
            A(nextStateIx,i) = 1;
        end
        
    elseif(s== 5) %middle guard
        
        if(c == patternLen)
            
            for ss = [16:25] %the next states can only be either starting quiet zone, or the starting guard
                s_next = ss;
                c_next = 1;
                m_next = 1;
                
                nextStateIx = StatesInv(c_next,s_next,m_next);
                A(nextStateIx,i) = (1/10);
            end
            
        else
            c_next = c+1;
            s_next = s;
            m_next = m;
            
            nextStateIx = StatesInv(c_next,s_next,m_next);
            A(nextStateIx,i) = 1;
        end
        
    elseif(s>= 6 && s<=15) %left symbols
        
        if(m~=6) %
            if(c == patternLen)

                for ss = [6:15] %the next states can only be either starting quiet zone, or the starting guard
                    s_next = ss;
                    c_next = 1;
                    m_next = m+1;

                    nextStateIx = StatesInv(c_next,s_next,m_next);
                    A(nextStateIx,i) = (1/10);
                end

            else
                c_next = c+1;
                s_next = s;
                m_next = m;

                nextStateIx = StatesInv(c_next,s_next,m_next);
                A(nextStateIx,i) = 1;
            end
        else %
            if(c == patternLen)
                s_next = 5;
                c_next = 1;
                m_next = 1;

                nextStateIx = StatesInv(c_next,s_next,m_next);
                A(nextStateIx,i) = 1;

            else
                c_next = c+1;
                s_next = s;
                m_next = m;

                nextStateIx = StatesInv(c_next,s_next,m_next);
                A(nextStateIx,i) = 1;
            end
        end  %
            
    elseif(s>= 16 && s<=25) %right symbols
        
        if(m~=6) %
            if(c == patternLen)

                for ss = [16:25] %the next states can only be either starting quiet zone, or the starting guard
                    s_next = ss;
                    c_next = 1;
                    m_next = m+1;

                    nextStateIx = StatesInv(c_next,s_next,m_next);
                    A(nextStateIx,i) = (1/10);
                end

            else
                c_next = c+1;
                s_next = s;
                m_next = m;

                nextStateIx = StatesInv(c_next,s_next,m_next);
                A(nextStateIx,i) = 1;
            end
        else %
            if(c == patternLen)
                s_next = 4;
                c_next = 1;
                m_next = 1;

                nextStateIx = StatesInv(c_next,s_next,m_next);
                A(nextStateIx,i) = 1;

            else
                c_next = c+1;
                s_next = s;
                m_next = m;

                nextStateIx = StatesInv(c_next,s_next,m_next);
                A(nextStateIx,i) = 1;
            end
        end  %
        
    else
        error('Unknown State!');
    end
end

%% Part2: Compute the inital probability

p_init = zeros(NumStates,1);

c = 1;
s = 1;
m = 1;

State_init = StatesInv(c,s,m);
p_init(State_init) = 1;

%the barcode *must* start with the "starting quite zone", with s_n=1. Other
%states are not possible. Fill the initial probability accordingly. 




%% Part3: Compute the log-likelihood
mu0=20;
mu1=250;
T = length(obs);

logObs = zeros(NumStates,T);

mu= [mu0 mu1]';

sigma = sqrt(estimated_obs_noise);

for t=1:T
    for j=1:NumStates
        logObs(j,t)=log(normpdf(obs(t),mu(f_kst(j)+1),sigma));
    end
    % you can use the variable f_kst here
end

% plot du signale xn
%figure;
%plot(obs)

%% part 3.5: generate the HMM 

T=123;

c_sim = zeros(1,T);
s_sim = zeros(1,T);
m_sim = zeros(1,T);
State_sim = zeros(1,T);
obs_sim = zeros(1,T);

c_sim(1)=1;
s_sim(1)=1;
m_sim(1)=1;
State_sim(1) = StatesInv(c_sim(1),s_sim(1),m_sim(1));
obs_sim(1)=normrnd(mu(f_kst(State_sim(1))+1),sigma);

for t=2:T
    next=find(A(:,State_sim(t-1))>0);
    State_sim(t)=next(randi(length(next)));
    c_sim(t)=States(State_sim(t),1);
    s_sim(t)=States(State_sim(t),2);
    m_sim(t)=States(State_sim(t),3);
    obs_sim(t)=normrnd(mu(f_kst(State_sim(t))+1),sigma);
end



%% part 3.51: observe the generated HMM 
obs_sim(obs_sim<0) = 0;
obs_sim(obs_sim>255) = 255;

ix = find(c_sim ==1);
s_ix = s_sim(ix);
code_sim = [];
for i = 1:length(s_ix)
    tmp = s_ix(i);
    %consider only the symbols that correspond to digits
    if(tmp>=6)
        chr = mod(tmp-6,10);
        code_sim = [code_sim, chr];
    end
    
end

%bar-code symbols
figure;
plot(obs_sim)
title('Simulation of signal')
xlabel('bar-code bytes')
ylabel('Observed intensity')

%Real bar-code symbols
figure;
plot(obs)
title('Actual signal')
xlabel('bar-code bytes')
ylabel('Observed intensity')

% bar-code image
bc_image = uint8((255- repmat(obs_sim, [100 1])));
figure;
imshow(bc_image)
set(gcf,'Position',[100 100 1000 500]);
title(num2str(code_sim));
xlabel('simulated bar-code');



%% Part 4: Compute the filtering distribution via Forward recursion
log_alpha = zeros(NumStates,T);
log_alpha_predict = zeros(NumStates,T);
prob_filter = zeros(NumStates,T);

for t=1:T
    if t==1
        log_alpha_predict(:,t) = log(p_init);
    else
        log_alpha_predict(:,t) = fw_predict(A,log_alpha(:,t-1));
        
    end;
    
    log_alpha(:,t) = fw_bw_update(logObs(:,t),log_alpha_predict(:,t));
    
end

%% Part 4.5: Computing the filtering distribution 

for t=1:T
   mx= max(log_alpha(:,t));
   prob_filter(:,t) = exp(log_alpha(:,t) - mx - log(sum(exp(log_alpha(:,t)-mx))));
end


%% Part 5: Compute via Forward-Backward recursion

log_beta= zeros(NumStates,T);
log_beta_postdict = zeros(NumStates,T);
%log_filter = zeros(NumStates,T);

for t=T:-1:1
    if t==T
        log_beta_postdict(:,t) = zeros(NumStates,1);
    else
        log_beta_postdict(:,t) = bwd_postdict(A,log_beta(:,t+1));
        
    end;
    
    log_beta(:,t) = fw_bw_update(logObs(:,t),log_beta_postdict(:,t));
    
end


%% Part 5.5  the smoothing distribution

log_gamma= log_alpha + log_beta_postdict;
smoothing =  zeros(NumStates,T);

% Computing the filtering distribution 

for t=1:T
   mx= max(log_gamma(:,t));
   smoothing(:,t) = exp(log_gamma(:,t) - mx - log(sum(exp(log_gamma(:,t)-mx))));
end


markov_states=zeros(3,T);
path_max=zeros(1,T);
for t=1:T
    [tmp,path_max(t)]=max(smoothing(:,t));
    markov_states(1,t)=States(path_max(t),1);
    markov_states(2,t)=States(path_max(t),2);
    markov_states(3,t)=States(path_max(t),3);
end

%plot(markov_states(1,:))
%plot(markov_states(2,:))
%plot(markov_states(3,:))


%% compute actual Sn


S_code=[ones(1,14) ones(1,3)*3] ;
for i=1:6
    S_code=[S_code ones(1,7)*(6+code(i))];
end
S_code=[S_code ones(1,5)*5];
for i=7:12
    S_code=[S_code ones(1,7)*(16+code(i))];
end
S_code=[S_code ones(1,3)*4 ones(1,14)*2];

%% Part 5.51 marginal smoothing distribution over Sn

Marginal_smoothing_S=zeros(S,T);

for t=1:T
    mx= max(log_gamma(:,t));
    for j=1:NumStates
        Marginal_smoothing_S(States(j,2),t)=exp(log_gamma(j,t)-mx)+Marginal_smoothing_S(States(j,2),t);
    end
end

figure;
imagesc(Marginal_smoothing_S)
colormap(flipud(gray))
hold on;
plot(markov_states(2,:),'red')
plot(S_code,'--')
legend('expected Sn','True Sn')
legend('Location','southwest')
title('Marginal smoothing distribution')
xlabel('bar-code bytes')
ylabel('Sn states')

%% Part 5.52: marginal filtering distribution
Marginal_filtering_S=zeros(S,T);

for t=1:T
    for j=1:NumStates
        Marginal_filtering_S(States(j,2),t)=prob_filter(j,t)+Marginal_filtering_S(States(j,2),t);
    end
end

figure;
imagesc(Marginal_filtering_S)
colormap(flipud(gray))
hold on;
plot(markov_states(2,:),'red')
plot(S_code,'--')
legend('expected Sn','True Sn')
legend('Location','southwest')
title('Marginal filtering distribution')
xlabel('bar-code bytes')
ylabel('Sn states')

%% Part 5.53 marginal smoothing distribution over Mn

M_code=[ones(1,14) ones(1,3)] ;
for i=1:6
    M_code=[M_code ones(1,7)*i];
end
M_code=[M_code ones(1,5)];
for i=7:12
    M_code=[M_code ones(1,7)*(i-6)];
end
M_code=[M_code ones(1,3) ones(1,14)];


Marginal_smoothing_M=zeros(M,T);

for t=1:T
    mx= max(log_gamma(:,t));
    for j=1:NumStates
        Marginal_smoothing_M(States(j,3),t)=exp(log_gamma(j,t)-mx)+Marginal_smoothing_M(States(j,3),t);
    end
end

figure;
imagesc(Marginal_smoothing_M)
colormap(flipud(gray))
hold on;
plot(markov_states(3,:),'red')
plot(M_code,'--')
legend('expected M_n','True M_n','Interpreter','latex')
legend('Location','southwest')
title('Marginal smoothing distribution')
xlabel('bar-code bytes')
ylabel('M_n states','Interpreter','latex')

%% Part 5.54: marginal filtering distribution of Mn
Marginal_filtering_M=zeros(M,T);

for t=1:T
    for j=1:NumStates
        Marginal_filtering_M(States(j,3),t)=prob_filter(j,t)+Marginal_filtering_M(States(j,3),t);
    end
end

figure;
imagesc(Marginal_filtering_M)
colormap(flipud(gray))
hold on;
plot(markov_states(3,:),'red')
plot(M_code,'--')
legend('expected M_n','True M_n','Interpreter','latex')
legend('Location','southwest')
title('Marginal filtering distribution')
xlabel('bar-code bytes')
ylabel('M_n states','Interpreter','latex')

%% Part 5.55 marginal smoothing distribution over Cn

C_code=[1:7 1:7 1:3] ;
for i=1:6
    C_code=[C_code 1:7];
end

C_code=[C_code 1:5];

for i=7:12
    C_code=[C_code 1:7];
end

C_code=[C_code 1:3 1:7 1:7];


Marginal_smoothing_C=zeros(C,T);

for t=1:T
    mx= max(log_gamma(:,t));
    for j=1:NumStates
        Marginal_smoothing_C(States(j,1),t)=exp(log_gamma(j,t)-mx)+Marginal_smoothing_C(States(j,1),t);
    end
end

figure;
imagesc(Marginal_smoothing_C)
colormap(flipud(gray))
hold on;
plot(markov_states(1,:),'red')
plot(C_code,'--')
legend('expected C_n','True C_n','Interpreter','latex','FontSize',14)
legend('Location','southwest')
title('Marginal smoothing distribution')
xlabel('bar-code bytes')
ylabel('C_n states','Interpreter','latex','FontSize',14)

%% Part 5.56: marginal filtering distribution of Cn
Marginal_filtering_C=zeros(C,T);

for t=1:T
    for j=1:NumStates
        Marginal_filtering_C(States(j,1),t)=prob_filter(j,t)+Marginal_filtering_C(States(j,1),t);
    end
end

figure;
imagesc(Marginal_filtering_C)
colormap(flipud(gray))
hold on;
plot(markov_states(1,:),'red')
plot(C_code,'--')
legend('expected C_n','True C_n','Interpreter','latex')
legend('Location','southwest')
title('Marginal filtering distribution')
xlabel('bar-code bytes')
ylabel('C_n states','Interpreter','latex','FontSize',14)

%% Part 6: Compute the most-likely path via Viterbi algorithm

T = length(obs);

nu = zeros(NumStates,T);
paths = zeros(NumStates,T);

nu(:,1)=log(p_init)+logObs(:,1);

path_max_V=ones(1,T);

for t=2:T
    for i=1:NumStates
        [tmpi,tmpj]=max(log(A(i,:))+nu(:,t-1)');
        nu(i,t)=tmpi+logObs(i,t);
        paths(i,t)=tmpj;
    end
end

[tmpi,path_max_V(T)]=max(nu(:,T));

%path_max_V(T)=745;
%reconstruire le chemin avec paths en reculant
for t=(T-1):-1:1
    path_max_V(t)=paths(path_max_V(t+1),t+1);
end

markov_states_V=zeros(3,T);
for t=1:T
    markov_states_V(1,t)=States(path_max_V(t),1);
    markov_states_V(2,t)=States(path_max_V(t),2);
    markov_states_V(3,t)=States(path_max_V(t),3);
end


%plot(markov_states_V(1,:))
%plot(markov_states_V(2,:))
%plot(markov_states_V(3,:))

figure;
hold on;
plot(markov_states(2,:),'red')
plot(S_code,'--')
legend('expected Sn','True Sn')
legend('Location','northwest')
title('Most likely path VS actual path')
xlabel('bar-code bytes')
ylabel('Sn states')

%% Part 7: Obtain the barcode string from the decoded states

best_cn = markov_states_V(1,:); %(this will be obtained via Viterbi)
best_sn = markov_states_V(2,:); %(this will be obtained via Viterbi)

%find the place where a new symbol starts
ix = find(best_cn ==1);
s_ix = best_sn(ix);
decoded_code = [];
for i = 1:length(s_ix)
    tmp = s_ix(i);
    %consider only the symbols that correspond to digits
    if(tmp>=6)
        chr = mod(tmp-6,10);
        decoded_code = [decoded_code, chr];
    end
    
end

fprintf('Real code:\t');
fprintf('%d',code);
fprintf('\n');
fprintf('Decoded code:\t');
fprintf('%d',decoded_code);
fprintf('\n');
