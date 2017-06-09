%% Initialization
clear ; close all; clc


%% ======================= Plotting data =======================
fprintf('Plotting Data ...\n')
data = load(fullfile('..', 'datasets','linearR_train.data'));

%data = load('datasets/linearR_train.data');
X = data(:, 1); y = data(:, 2);
m = length(y); % number of training examples

plotData(X, y);

%% =================== Plotting theta ===================

% Plot the linear fit

hold on; % keep previous plot visible

thetas = load(fullfile('..', 'results','linear_regressoin_gradient_descent_thetas.data'));

t0 = thetas(:, 1);
t1 = thetas(:, 2);

h = t1 .* X + t0;

plot(X(:,1), h, '-')

