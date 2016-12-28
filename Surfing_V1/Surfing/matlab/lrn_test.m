format long
beta = 0.75;
K = 2.0;
N = 5;
alpha = 1e-4;

A = [1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 4, 7, 2, 5, 8, 3;
6, 9, 2, 2, 3, 4, 6, 6, 7, 8, 10,1, -2, -4, 2, 1 ];
B =  [0.1, 0.2, 0.3, 0.4, 0, 5, 0, 6, 0.8, 0.3, 0.1, 0.2, 0.3, 0.4, 0, 5;
    0, 6,0.8, 0.3, 0.1, 0.2, 0.3, 0.4, 0, 5, 0, 6, 0.8, 0.3, 0.3, 0.7];


for i = 1:16
    AA(i) = A(1,i)*(2 + alpha*(A(1,i) + A(2,i)))^-beta;
end

for i = 1:16
    AA(i+16) = A(2,i)*(2 + alpha*(A(1,i) + A(2,i)))^-beta;
end

for i = 1:16
    BB(i) =  (B(1,i))*((2 + alpha*(A(1,i) + A(2,i)))^-beta - alpha *A(1,i)*(2 + alpha*(A(1,i) + A(2,i)))^-(1+beta))...
    - B(2,i)*alpha *A(2,i)*(2 + alpha*(A(1,i) + A(2,i)))^-(1+beta) ;
end

for i = 1 : 16
    BB(16+i) =  (B(2,i))*((2 + alpha*(A(1,i) + A(2,i)))^-beta - alpha *A(2,i)*(2 + alpha*(A(1,i) + A(2,i)))^-(1+beta))...
    - B(1,i)*alpha *A(1,i)*(2 + alpha*(A(1,i) + A(2,i)))^-(1+beta) ;
end