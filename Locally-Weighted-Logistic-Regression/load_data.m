function [X,y] = load_data

X = load('data/x.dat');
y = load('data/y.dat');
%plot_lwlr(X, y, 0.5, 0.3);