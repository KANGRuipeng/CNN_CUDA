A = [0.3 0.2 0.3 0.1 0.5 0.6];
B = zeros(1,6);

S = 0;

for i = 1 : 6
    S = S + exp(A(i));
end

for i = 1 : 6
   B(i) = exp(A(i)) /S;
end

C = [0 0 1 0 0 0];

D = B - C;

E = zeros(1,6);

for i = 1 : 6
    for j = 1 : 6
       if i == j
           E(i) = E(i) + B(j)*(1 - B(j))*D(j);
       else
           E(i) = E(i) - B(j)*B(i)*D(j);
       end
    end
end