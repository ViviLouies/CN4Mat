% B=flip(A,dim)
% previous version: flipdim
% A表示一个矩阵，dim指定翻转方式。
% dim为1，表示每一列进行逆序排列；
% dim为2，表示每一行进行逆序排列。
function X = rot180(X)
X = flip(flip(X, 1), 2);
end