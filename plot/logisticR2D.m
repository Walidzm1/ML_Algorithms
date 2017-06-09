%% Initialization
clear ; close all; clc


%% ======================= Plotting data =======================
fprintf('Plotting Data ...\n')
data = load(fullfile('..', 'datasets','logisticR_train.data'));

%data = load('datasets/linearR_train.data');
X = data(:, 1:2); y = data(:, 2);

m = length(y); % number of training examples

plotData(X, y);

%% =================== Plotting theta ===================

% Plot the linear fit

hold on; % keep previous plot visible

thetas = load(fullfile('..', 'results','logistic_regressoin_gradient_descent_thetas.data'));

t0 = thetas(:, 1);
t1 = thetas(:, 2);
t2 = thetas(:, 3);


tmp = ((-1 * t0) - t1 .* X(:,1)) / t2;


%h = t0 + t1 .* X(:,1) + t2 .* X(:,2);

plot(X(:,1), tmp, '-')

