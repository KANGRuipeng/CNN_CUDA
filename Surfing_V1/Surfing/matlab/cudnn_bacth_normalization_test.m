clear all
clc

A = [ 1, 2, 3; 2, 5, 4; -2, 8, 2 ];
B = [ 0.1, 0.2, -0.2; 0.3, 0.5, -0.2; -0.2, 0.3, 0.7];

[row,clo] = size(A);

Sum = zeros(1,clo);
Mean = zeros(1,clo);

gamma = ones(1,clo);
beta = zeros(1,clo);

for i = 1 : clo
    for j = 1 : row
        Sum(i) = Sum(i) + A(j,i);
    end
    Mean(i) = Sum(i) / row;
end

Var_Sum = zeros(1, clo);
Var = zeros(1, clo);
Var_Inv = zeros(1,clo);

for i = 1 : clo
    for j = 1 : row
        Var_Sum(i) = Var_Sum(i) + (A(j,i)-Mean(i))^2;
    end
    Var(i) = Var_Sum(i) / (row - 1);
    Var_Inv(i) = 1 /(sqrt(Var_Sum(i)/row));
end

for i = 1: clo
    for j = 1: row
        x_hat(j,i) = (A(j,i) - Mean(i))*Var_Inv(i);
        y(j,i) = gamma(i)* x_hat(j,i) + beta(i);
    end 
end


d_gamma = zeros(1, clo);
for i = 1: clo
    for j = 1: row
        d_gamma(i) = d_gamma(i) + B(j,i) * x_hat(j,i);
    end 
end

d_beta = zeros(1,clo);
for i = 1: clo
    for j = 1: row
        d_beta(i) = d_beta(i) + B(j,i);
    end 
end

d_var = zeros(1, clo);
d_mean = zeros(1, clo);

for i = 1: clo
    for j = 1: row
        d_var(i) = d_var(i) + B(j,i)*gamma(i)*(A(j,i) -Mean(i))*(-0.5)*Var_Inv(i)^(3);
        d_mean(i) = d_mean(i) + B(j,i)*(-1)*Var_Inv(i);
    end 
end

for i = 1: clo
    for j = 1: row
        d_x(j,i) =  B(j,i)*gamma(i)*Var_Inv(i) + d_var(i)*2*(A(j,i) -Mean(i))/row + d_mean(i)/row;
    end 
end



