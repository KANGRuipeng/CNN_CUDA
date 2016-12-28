% This file is test and compare with CUDNN lib, may work as reference for
% future use
A1 = [1 2 3 ; 4 5 6; 7 8 9];
A2 = A1';
A3 = A1 + eye(3);
A4 = A3 -A2;

B1 = [1 2 ; 3 4];
B2 = [3,4 ; 5 6];
B3 = B2 + 2*eye(2);
B4 = B1 - 3*eye(2);
B5 = B2 - B4;
B6 = B3 + B1;

C1 = convn(A1, B1, 'valid');
C2 = convn(A2, B2, 'valid');
C3 = convn(A1, B3, 'valid');
C4 = convn(A2, B4, 'valid');
C5 = convn(A1, B5, 'valid');
C6 = convn(A2, B6, 'valid');
C7 = convn(A3, B1, 'valid');
C8 = convn(A4, B2, 'valid');
C9 = convn(A3, B3, 'valid');
C10 = convn(A4, B4, 'valid');
C11 = convn(A3, B5, 'valid');
C12 = convn(A4, B6, 'valid');

D11 = C1 + C2;
D12 = C3 + C4;
D13 = C5 + C6;
D21 = C7 + C8;
D22 = C9 + C10;
D23 = C11 + C12;

Error = [0.1 -0.1; 0.1, 0.2];

E1 = B1 + Error;
E2 = B2 + Error;
E3 = B3 + Error;
E4 = B4 + Error;
E5 = B5 + Error;
E6 = B6 + Error;


F1 = convn(A1, E1, 'valid');
F2 = convn(A2, E2, 'valid');
F3 = convn(A1, E3, 'valid');
F4 = convn(A2, E4, 'valid');
F5 = convn(A1, E5, 'valid');
F6 = convn(A2, E6, 'valid');
F7 = convn(A3, E1, 'valid');
F8 = convn(A4, E2, 'valid');
F9 = convn(A3, E3, 'valid');
F10 = convn(A4, E4, 'valid');
F11 = convn(A3, E5, 'valid');
F12 = convn(A4, E6, 'valid');

G11 = F1 + F2;
G12 = F3 + F4;
G13 = F5 + F6;
G21 = F7 + F8;
G22 = F9 + F10;
G23 = F11 + F12;


EE1 = G11 - D11;
EE2 = G12 - D12;
EE3 = G13 - D13;
EE4 = G21 - D21;
EE5 = G22 - D22;
EE6 = G23 - D23;


H1 = convn(EE1, Rot180(B1), 'full');
H2 = convn(EE1, Rot180(B2), 'full');
H3 = convn(EE2, Rot180(B3), 'full');
H4 = convn(EE2, Rot180(B4), 'full');
H5 = convn(EE3, Rot180(B5), 'full');
H6 = convn(EE3, Rot180(B6), 'full');
H7 = convn(EE4, Rot180(B1), 'full');
H8 = convn(EE4, Rot180(B2), 'full');
H9 = convn(EE5, Rot180(B3), 'full');
H10 = convn(EE5, Rot180(B4), 'full');
H11 = convn(EE6, Rot180(B5), 'full');
H12 = convn(EE6, Rot180(B6), 'full');

J1 = H1 + H3 + H5;
J2 = H2 + H4 + H6;
J3 = H7 + H9 + H11;
J4 = H8 + H10 + H12;

K1 = Rot180(conv2( A1,Rot180(EE1) ,'valid' ));
K2 = Rot180(conv2( A2,Rot180(EE1) ,'valid' ));
K3 = Rot180(conv2( A1,Rot180(EE2) ,'valid' ));
K4 = Rot180(conv2( A2,Rot180(EE2) ,'valid' ));
K5 = Rot180(conv2( A1,Rot180(EE3) ,'valid' ));
K6 = Rot180(conv2( A2,Rot180(EE3) ,'valid' ));
K7 = Rot180(conv2( A3,Rot180(EE4) ,'valid' ));
K8 = Rot180(conv2( A4,Rot180(EE4) ,'valid' ));
K9 = Rot180(conv2( A3,Rot180(EE5) ,'valid' ));
K10 = Rot180(conv2( A4,Rot180(EE5) ,'valid' ));
K11 = Rot180(conv2( A3,Rot180(EE6) ,'valid' ));
K12 = Rot180(conv2( A4,Rot180(EE6) ,'valid' )); 

% A1 and A3 use information of B1 ,which is related to EE1 and EE4, then
L1 = K1 + K7;
L2 = K2 + K8;
L3 = K3 + K9;
L4 = K4 + K10;
L5 = K5 + K11;
L6 = K6 + K12;
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 