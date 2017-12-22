clear
close all
clc

%% Data generation
F = 10;
T = 20; 
K = 2;

Wreal = 5*rand(F,K);
Wreal(rand(F*K,1)<0.6) = 0.0001;

Hreal = 5*rand(K,T);
Hreal(rand(K*T,1)<0.6) = 0.0001;

V = Wreal*Hreal;
V = poissrnd(V);
V(V <eps) = eps; 


%% Multiplicative update rules (MUR)


W = 5*rand(F,K);
H = 5*rand(K,T);

O = ones(size(V));

MaxIter = 50;
err = zeros(MaxIter,1);

figure;
pause
for i = 1:MaxIter
    
    % Updating W
    Vhat = W*H;
    Vhat = Vhat + eps; %for numerical stability
    
    err(i) = sum(sum(V.*log(V) - V.*log(Vhat) - V + Vhat));
    
    W = W .* (((V./Vhat)*H') ./ (O*H'));
    
    % Updating H
    Vhat = W*H;
    Vhat = Vhat + eps; %for numerical stability
    
    H = H .* (( W'*(V./Vhat)) ./ (W'*O));
    
    subplot(3,3,[2 3]);
    imagesc(H); axis xy;
    colorbar;
    title('H');
    
    subplot(3,3,[4 7]);
    imagesc(W); axis xy;
    colorbar;
    title('W');
    
    subplot(3,3,[5 6 8 9]);
    imagesc(V); axis xy;
    colorbar;
    title('V');
    
    
    drawnow
    
    
    
    disp(i);
end
%%

figure, semilogy(err);
xlabel('Iterations')
ylabel('Error');





