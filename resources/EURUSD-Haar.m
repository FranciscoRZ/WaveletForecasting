clc;
clear;
close all;

%%
[eurodol] = xlsread("../data/eurusd.xlsx");
eurodol = eurodol(:,2);

%%
% Param�tre
j = -4;

%
nb_eurodol = size(eurodol, 1);
t = transpose(linspace(0, nb_eurodol - 1, nb_eurodol));
k =  transpose(linspace(-1, nb_eurodol - 2, nb_eurodol));

%%
% D�termination de nb_k
nb_k = 1;
while (k(nb_k, 1) * 2^(-j)) < nb_eurodol
    nb_k = nb_k + 1;
end
k = k(1:nb_k, 1);

%% 

% Initialisations
k1 = zeros(nb_k + 1, 1);
k2 = zeros(nb_k + 1, 1);
coeff = zeros(nb_k, 1);

%%
% Calcul du coefficient d'�chelle : coeff
k1 = k.*2^(-j);
k2 = (k+1).*2^(-j);
for i = 1:nb_k
    
    dummy1 = t >= k1(i, 1);
    dummy2 = t >= k2(i, 1);
    
    sum1 = sum(eurodol.*dummy1);
    sum2 = sum(eurodol.*dummy2);
    
    coeff(i, 1) = 2^(j/2) * (sum1 - sum2);
end

%%
% Calcul de l'approximation : approx
t1 = zeros(nb_eurodol - 1, 1);
t2 = zeros(nb_eurodol - 1, 1);
approx = zeros(nb_eurodol, 1);

a = 0;
b = 0;

for i = 2:nb_eurodol
    t1(i-1, 1) = 2^j * t(i-1, 1) - 1;
    t2(i-1, 1) = 2^j * t(i-1, 1);
    
    dummy1 = k >= t1(i-1, 1);
    dummy2 = k >= t2(i-1, 1);
    
    sum1 = sum(coeff.*dummy1);
    sum2 = sum(coeff.*dummy2);
    
    approx(i, 1) = 2^(j/2) * (sum1 - sum2);
end

%%
approx = approx(3:nb_eurodol - 2^(-j));
eurodol = eurodol(3:nb_eurodol - 2^(-j));
coeff_display = coeff(2:size(coeff,1)-2);

%%
% Affichage graphique
subplot(2,1,1)
hold on
title("Scale coefficients")
plot(coeff_display)
hold off

subplot(2,1,2)
hold on
title("EURUSD approximation")
plot(approx)
%plot(eurodol)
hold off
