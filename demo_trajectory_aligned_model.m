% This code is an implementation of the Extended Target Tracking approach
% described in:  
% Şahin, K.K., Balcı, A.E., Özkan, E.: Random matrix extended target tracking 
% for trajectory-aligned and drifting targets. 
% IET Radar Sonar Navig. 18(11), 2247–2263 (2024). https://doi.org/10.1049/rsn2.12628

%% Clear Workspace and Figures
clear
close all

%% Simulation Parameters
% Set up the simulation length, sampling time, and other scenario parameters
num_of_frames = 60;    % Number of measurement batches (scans) in this simulation
T = 1.0;               % Time step in seconds
end_time = (num_of_frames - 1)*T; % Total simulation time in seconds
t = linspace(0, end_time, num_of_frames); % Time vector for the simulation
d = 2;                 % Dimension of the extent (2D)
R = 1e0*eye(d);        % Sensor noise covariance matrix (measurement noise)
H = kron([1 0], eye(d)); % Measurement matrix extracting position from state
mean_num_of_meas = 15; % Average number of measurements per scan
I_max = 10;            % Maximum number of Variational Bayes iterations for measurement update

%% Initial Parameters for the State and Extent
% The target state is parameterized as [x; y; vx; vy; theta; vtheta]
% where x, y: position; vx, vy: velocity; theta: orientation; vtheta: orientation rate.

state = [0; 0; 10; 0; 0; 0];  % Initial true state of the target
extent = [25/4 0; 0 1];       % True extent (elliptical shape matrix)
[V,D] = eig(extent);          % Eigen-decomposition of extent for plotting

Q = 5*eye(d);                 % Process noise for linear motion states

% Initial estimates for state and covariance (for the filter)
x_k_k_minus_1_cart = [0; 0; 0; 0];   % Initial state estimate in Cartesian form [x; y; vx; vy]
P_k_k_minus_1_cart = eye(4)*10;      % Initial covariance for state estimate

% For heading angles and orientation:
theta_k_k_minus_1 = 1e-3;    % Initial orientation angle estimate
Theta_k_k_minus_1 = 1;       % Initial orientation angle variance
thetadot_k_k_minus_1 = 1e-3; % Initial angular velocity estimate
Thetadot_k_k_minus_1 = 1;    % Angular velocity variance
ThetaQ = 1e-2;               % Process noise of the orientation angle model

% Inverse-Gamma parameters for the extent distribution:
% alpha and beta define the shape and scale parameters of Inverse-Gamma distributions
alpha_k_k_minus_1 = ones(d,1)*3; 
beta_k_k_minus_1 = ones(d,1)*10; 

% Initial expected extent matrix
EX_k_k_minus_1 = 5*eye(d); 

% Forgetting factor parameter tau for exponential forgetting (used in VB updates)
tau = 10*T;  
s = 0.25;   % Scale parameter used in the VB update equations

% Construct the full initial state vector that includes orientation terms
% Here we assume the orientation state is appended to the basic Cartesian state:
% For the trajectory-aligned model, we may consider a reduced state vector.
% Adjust as per the provided measurement/time update functions.
x_k_k_minus_1 = [x_k_k_minus_1_cart(1:3); theta_k_k_minus_1; thetadot_k_k_minus_1]; 

% Construct the initial covariance matrix including orientation terms
P_k_k_minus_1 = blkdiag(P_k_k_minus_1_cart(1:3, 1:3), Theta_k_k_minus_1, Thetadot_k_k_minus_1);

%% Generate Measurements
% Generate synthetic measurements for the scenario using a provided demo function.
% The function "ettGenerateDemoAlignedScenario" simulates measurements from the 
% trajectory-aligned and drifting target model described in the reference.
[ett_measurements, ett_ground_truth] = ettGenerateDemoAlignedScenario(t, state, extent, H, R, mean_num_of_meas);

%% Initialize Plotter
set(groot, 'DefaultLegendInterpreter', 'latex'); % Use LaTeX interpreter for better formatting
f = figure('units','normalized','outerposition',[0 0 1 1]); % Full-screen figure
% Set plotting axes around the target scenario:
axis([-200 280 -50 280]*0.5 + kron(-state(1:2)', [-1 -1]));
grid on; hold all;
colors = [198,0,1; 0,255,0; 0,255,255]/256; % Define colors for plotting
legend on; legend('Location','northwest');

%% Main Filtering Loop
% For each scan k, we do:
% 1. Measurement Update (VB-based update from new measurements)
% 2. Time Update (Predict next state and extent distribution)
% 3. Visualization of results

for k=1:num_of_frames

    % Extract the measurements at time step k
    Y_k = ett_measurements{k};

    % Measurement Update (Correction step)
    % Use the provided "trajectoryAlignedModelMeasurementUpdate" function that applies
    % the VB update equations for the trajectory-aligned extended target model.
    [x_k_k(:,k), P_k_k(:,:,k), alpha_k_k(:,k), beta_k_k(:,k), EX_k_k(:,:,k)] ...
        = trajectoryAlignedModelMeasurementUpdate(x_k_k_minus_1(:,k), P_k_k_minus_1(:,:,k), ...
        alpha_k_k_minus_1(:,k), beta_k_k_minus_1(:,k), s, R, Y_k, I_max);

    % Time Update (Prediction step)
    % Propagate the state and extent parameters forward one time step.
    [x_k_k_minus_1(:,k+1), P_k_k_minus_1(:,:,k+1), alpha_k_k_minus_1(:,k+1), ...
        beta_k_k_minus_1(:,k+1)] = trajectoryAlignedModelTimeUpdate(x_k_k(:,k), ...
        P_k_k(:,:,k), alpha_k_k(:,k), beta_k_k(:,k), diag([Q(1,1); ThetaQ]), T, tau);

    % Plotting the current scenario:
    % Measurements:
    measurement_plot = scatter(Y_k(1,:), Y_k(2,:), 20, '*k'); 
    
    % Ground truth extent:
    gt_extent = drawEllipse(ett_ground_truth.extents(:,:,k+1), ett_ground_truth.states(1:2,k+1), 1);
    gt_extent.LineWidth = 2;
    gt_extent.Color = "k";
    
    % Estimated extent:
    est_extent = drawEllipse(EX_k_k(:, :, k), x_k_k(1:2, k), 1);
    est_extent.Color = colors(3,:);
    est_extent.LineStyle = "--";
    est_extent.LineWidth = 2;

    % Update legend:
    Legend = {'Measurements', 'Ground Truth', 'Estimation (P1)'};
    pause(1e-3); % Small pause to visualize the updates
    legend(Legend);

end

%% Compute Error Metrics - Gauss-Wasserstein Distance
% After all scans, compute performance metrics such as the Gauss-Wasserstein (GW) distance,
% which measures how well the estimated distribution matches the ground truth distribution.
P1_results_state = zeros(1, num_of_frames);
P1_results_extent = zeros(1, num_of_frames);

for k=1:num_of_frames
    [P1_results_state(k), P1_results_extent(k)] = gwDistance( ...
        ett_ground_truth.states(1:2, k+1), x_k_k(1:2, k), ...
        ett_ground_truth.extents(:, :, k+1), EX_k_k(:, :, k));
end

GW_estimation = mean(sqrt(P1_results_state + P1_results_extent)) % Average GW distance

% RMSE of the orientation angle:
squared_error_heading = (wrapToPi(x_k_k(4,:) - ett_ground_truth.states(5,2:end))).^2;
RMSE_theta_estimation = sqrt(mean(squared_error_heading))

%% Helper Functions

function plot_handle = drawEllipse(M, center, n)
% drawEllipse:
%   Draws an ellipse defined by a positive-definite matrix M, centered at 'center',
%   scaled by factor n.
%
% Inputs:
%   M: 2x2 covariance or shape matrix defining the ellipse
%   center: [x; y] center of the ellipse
%   n: scaling factor (e.g., 1 for 1-sigma ellipse)
%
% Output:
%   plot_handle: handle to the generated plot object

    [V,D] = eig(M); % Eigen decomposition to get principal axes
    t = linspace(0, 2*pi, 100); % Parametric angle
    ellipse = V * sqrt(D) * [n*cos(t); n*sin(t)] + center; 
    plot_handle = plot(ellipse(1,:), ellipse(2,:));
end

function [center_term, extent_term] = gwDistance(m1, m2, X1, X2)
% gwDistance:
%   Computes the squared Gauss-Wasserstein distance between two Gaussian distributions 
%   defined by means (m1, X1) and (m2, X2).
%
% The GW distance is defined as:
%   GW^2 = ||m1 - m2||^2 + trace(X1 + X2 - 2*(X1^(1/2)*X2*X1^(1/2))^(1/2))
%
% Inputs:
%   m1, m2: mean vectors (2D)
%   X1, X2: covariance matrices (2x2)
%
% Outputs:
%   center_term: squared distance between means ||m1 - m2||^2
%   extent_term: covariance-related term trace(X1 + X2 - 2*(X1^(1/2)*X2*X1^(1/2))^(1/2))

    center_term = norm(m1 - m2)^2;
    extent_term = trace(X1 + X2 - 2*(X1^(1/2) * X2 * X1^(1/2))^(1/2));
end
