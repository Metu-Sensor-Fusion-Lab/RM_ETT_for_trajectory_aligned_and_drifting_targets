% This code is an implementation of the Extended Target Tracking approach
% described in:  Şahin, K.K., Balcı, A.E., Özkan, E.: Random matrix
% extended target tracking for trajectory-aligned and drifting targets. IET
% Radar Sonar Navig. 18(11), 2247–2263 (2024). https://doi.org/10.1049/rsn2.12628  

%% Clear workspace and figures
clear
close all

%% Simulation Parameters
num_of_frames = 60;   % Number of measurement batches (scans) in the simulation
T = 1.0;              % Timestep in seconds
end_time = (num_of_frames - 1)*T; 
t = linspace(0, end_time, num_of_frames); % Time vector
d = 2;                % Extent dimension (2D ellipse)
R = 1e0*eye(d);       % Measurement noise covariance
H = kron([1 0], eye(d)); % Measurement matrix (extract position from state)
mean_num_of_meas = 15;   % Average number of measurements per scan
I_max = 10;           % Maximum number of VB (Variational Bayes) iterations

%% Initial Parameters for the Dynamic and Extent Model
% The state vector is defined as: [x; y; vx; vy; theta; omega_theta]
% where (x,y) are position, (vx, vy) velocity components, theta is orientation, and omega_theta is angular velocity.
state = [0; 0; 20; 0; 0; 0]; % Initial "true" state of the target
extent = [36 0; 0 9/4];      % True extent (shape) of the target (2D ellipse)
[V,D] = eig(extent);         % Eigen decomposition of the true extent (for plotting and simulation)

% Process noise for Cartesian states
Q = 1e1*eye(d); 

% Initial prior for the estimated state and covariance
x_k_k_minus_1_cart = [0; 0; 0; 0];        % Initial state estimate in Cartesian form [x; y; vx; vy]
P_k_k_minus_1_cart = eye(4)*10;           % Initial state covariance

% Initial orientation and angular velocity priors
theta_k_k_minus_1 = 1e-3;     % Initial orientation angle prior
Theta_k_k_minus_1 = 1;        % Initial orientation angle variance
thetadot_k_k_minus_1 = 1e-3;  % Initial angular velocity prior
Thetadot_k_k_minus_1 = 1;     % Initial angular velocity variance
ThetaQ = 1e-2;                % Process noise for orientation angle dynamics
% Initial parameters for the Inverse-Gamma distribution governing the extent
alpha_k_k_minus_1 = ones(d,1)*3;  % shape parameters
beta_k_k_minus_1 = ones(d,1)*10;  % scale parameters

% Initial expected extent matrix
EX_k_k_minus_1 = 5*eye(d); 

% Forgetting factor parameter (tau) for the VB update (see Section 3.2 in the referenced paper)
tau = 10*T; 
s = 0.25;    % Scaling parameter used in the paper (see equations related to E[(sX)^{-1}])

%% Combine Orientation State and Covariances
x_k_k_minus_1 = x_k_k_minus_1_cart;
P_k_k_minus_1 = P_k_k_minus_1_cart;

% Orientation vector and covariance in a joint form: [theta; theta_dot]
theta_k_k_minus_1 = [theta_k_k_minus_1; thetadot_k_k_minus_1];
Theta_k_k_minus_1 = diag([Theta_k_k_minus_1; Thetadot_k_k_minus_1]);

%% Generate Measurements
% Uses a demonstration scenario generator which mimics a drift model (e.g., a target turning).
rng(2);
[ett_measurements, ett_ground_truth] = ettGenerateDemoDriftScenario(t, state, extent, H, R, mean_num_of_meas);

%% Visualization Setup
plotting = 1;
set(groot, 'DefaultLegendInterpreter', 'latex')
f = figure('units','normalized','outerposition',[0 0 1 1]);
axis([-20 300 -20 400]);
grid on; hold on;
colors = [198,0,1; 0,255,0; 0,255,255]/256;
legend('Location','northwest');

%% Main Filtering Loop
% In each iteration, we perform a measurement update followed by a time update.
% The measurement update step uses variational Bayes to update the state and
% extent distributions based on received measurements, as described in the
% referenced paper (see Section 3.2 and 3.3 and their subsections).

for k=1:num_of_frames
    % Get the measurements at time-step k
    Y_k = ett_measurements{k};

    % Measurement Update (Variational Bayes)
    [x_k_k(:,k), P_k_k(:,:,k), alpha_k_k(:,k), beta_k_k(:,k), theta_k_k(:,k), ...
        Theta_k_k(:,:,k), EX_k_k(:,:,k)] =  driftModelMeasurementUpdate(...
        x_k_k_minus_1(:,k), P_k_k_minus_1(:,:,k), alpha_k_k_minus_1(:,k), ...
        beta_k_k_minus_1(:,k), theta_k_k_minus_1(:,k), Theta_k_k_minus_1(:,:,k), s, R, Y_k, I_max);

    % Time Update (Prediction Step)
    % Propagate the state, extent, and orientation distributions forward in time.
    % Reference: Equations in Section 3 (time update step) of the cited paper.
    [x_k_k_minus_1(:,k+1), P_k_k_minus_1(:,:,k+1), alpha_k_k_minus_1(:,k+1), ...
        beta_k_k_minus_1(:,k+1), theta_k_k_minus_1(:,k+1), Theta_k_k_minus_1(:,:,k+1)] = ...
        driftModelTimeUpdate(x_k_k(:,k), P_k_k(:,:,k), alpha_k_k(:,k), beta_k_k(:,k), Q, ...
        theta_k_k(:,k), Theta_k_k(:,:,k), ThetaQ, T, tau);

    % Plotting the current estimates and ground truth
    if plotting
        measurement_plot = scatter(Y_k(1,:), Y_k(2,:), 20, '*k'); % Plot measurements
        gt_extent = drawEllipse(ett_ground_truth.extents(:,:,k+1), ett_ground_truth.states(1:2,k+1), 1);
        gt_extent.LineWidth = 2;
        gt_extent.Color = "k";
        
        est_extent = drawEllipse(EX_k_k(:, :, k), x_k_k(1:2, k), 1);
        est_extent.Color = colors(2,:);
        est_extent.LineStyle = "-";
        est_extent.LineWidth = 2;
        
        Legend = {'Measurements', 'Ground Truth', 'Estimation (Proposed Method)'};
        pause(1e-3);
        legend(Legend);
    end
end

%% Compute Error Metrics (Gauss-Wasserstein Distance)
% The Gauss-Wasserstein distance is used to measure how close the estimated
% Gaussian distribution is to the ground-truth distribution of the target state and extent.
% For details, see Section 4.2 of the referenced paper or related literature.

GW_results_state = zeros(1, num_of_frames);
GW_results_extent = zeros(1, num_of_frames);

for k=1:num_of_frames
    [GW_results_state(k), GW_results_extent(k)] = gwDistance(ett_ground_truth.states(1:2, k+1), x_k_k(1:2, k), ...
        ett_ground_truth.extents(:, :, k+1), EX_k_k(:, :, k));
end

% Compute the mean Gauss-Wasserstein distance over time
gw_mean = mean(sqrt(GW_results_state + GW_results_extent))

% Compute RMSE of orientation angle
% wrapToPi ensures angles remain within [-pi, pi]
theta_error = wrapToPi(theta_k_k(1,:) - ett_ground_truth.states(5,2:end));
orientation_RMSE = sqrt(mean(theta_error.^2))

%% Helper Functions

function plot_handle = drawEllipse(M, center, n)
% drawEllipse plots an ellipse defined by the positive definite matrix M,
% centered at 'center', scaled by a factor n.
%
% Inputs:
%  M      : positive definite matrix defining the ellipse
%  center : center coordinates of the ellipse [x; y]
%  n      : scaling factor
%
% Output:
%  plot_handle : handle to the plot object

    [V,D] = eig(M);
    t = linspace(0, 2*pi, 100);
    ellipse = V * sqrt(D) * [n*cos(t); n*sin(t)] + center;
    plot_handle = plot(ellipse(1,:), ellipse(2,:));
end

function [center_term, extent_term] = gwDistance(m1, m2, X1, X2)
% gwDistance computes the 2D Gauss-Wasserstein (GW) distance between two Gaussian distributions
% characterized by means (m1, m2) and covariances (X1, X2).
%
% Reference:
% Gauss-Wasserstein Distance is discussed in Section 4.2 of the referenced paper and
% in general literature on distribution distances.
%
% Inputs:
%  m1, m2 : mean vectors of the two Gaussians
%  X1, X2 : covariance matrices of the two Gaussians
%
% Outputs:
%  center_term : squared Euclidean distance between means
%  extent_term : trace-based term involving the covariances

    center_term = norm(m1 - m2)^2;
    extent_term = trace(X1 + X2 - 2*(X1^(1/2) * X2 * X1^(1/2))^(1/2));
end
