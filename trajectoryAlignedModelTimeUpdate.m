function [x_predicted, P_predicted, alpha_predicted, beta_predicted] = trajectoryAlignedModelTimeUpdate(x_k_k, P_k_k, alpha_k_k, beta_k_k, Q, T, tau)
% TRAJECTORYALIGNEDMODELTIMEUPDATE:
% Performs the time update (prediction) step for the trajectory-aligned model.
% This step propagates the state, covariance, and Inverse-Gamma parameters forward in time.
%
% Inputs:
%   x_k_k        : Updated state vector at time k
%   P_k_k        : Updated covariance matrix at time k
%   alpha_k_k    : Updated shape parameter of the Inverse-Gamma distribution
%   beta_k_k     : Updated scale parameter of the Inverse-Gamma distribution
%   Q            : Process noise covariance matrix for the dynamic model
%   T            : Time step (duration between updates)
%   tau          : Forgetting factor for the shape parmeters
%
% Outputs:
%   x_predicted  : Predicted state vector at time k+1
%   P_predicted  : Predicted covariance matrix at time k+1
%   alpha_predicted : Predicted shape parameter of the Inverse-Gamma distribution
%   beta_predicted  : Predicted scale parameter of the Inverse-Gamma distribution

    % Predict state and covariance using the constant turn-rate model (Equation 58)
    [x_predicted, F_dot_k, Q_k] = modelConstantTurnRate(x_k_k, Q, T);
    
    % Predict covariance matrix (Equation 59)
    P_predicted = F_dot_k * P_k_k * F_dot_k' + Q_k;

    % Update shape and scale parameters for Inverse-Gamma distribution (Equation 60)
    alpha_predicted = alpha_k_k * exp(-T / tau); % Shape parameter
    beta_predicted = beta_k_k * exp(-T / tau);   % Scale parameter
end

function [x_k_k_minus_1, F_dot, Q_k] = modelConstantTurnRate(x, Q, T)
% MODELCONSTANTTURNRATE:
% Implements the constant turn-rate (CTR) motion model for state prediction.
%
% Inputs:
%   x : Current state vector [x, y, v, h, w], where:
%       x, y : Position coordinates
%       v    : Velocity magnitude
%       h    : Heading (orientation angle)
%       w    : Turn rate (angular velocity)
%   Q : Process noise covariance matrix
%   T : Time step
%
% Outputs:
%   x_k_k_minus_1 : Predicted state vector at time k+1
%   F_dot         : State transition matrix (Jacobian of motion model)
%   Q_k           : Process noise covariance matrix at time k+1

    % Extract state components for readability
    x1 = x(1); % x-position
    x2 = x(2); % y-position
    v = x(3);  % Speed
    h = x(4);  % Heading angle
    w = x(5);  % Turn rate

    % Handle two cases: w (turn rate) close to zero or non-zero
    if abs(w) <= 1e-9
        % Case 1: Turn rate is approximately zero (straight-line motion)
        x_k_k_minus_1 = [x1 + v * T * cos(h); % New x-position
                         x2 + v * T * sin(h); % New y-position
                         v;                   % Velocity remains the same
                         h;                   % Heading remains the same
                         w];                  % Turn rate remains the same
        
        % State transition matrix for straight-line motion
        F_dot = [1, 0, T*cos(h), -v*T*sin(h), 0;
                 0, 1, T*sin(h),  v*T*cos(h), 0;
                 0, 0,         1,           0, 0;
                 0, 0,         0,           1, 0;
                 0, 0,         0,           0, 1];
    else
        % Case 2: Non-zero turn rate (curved motion)
        % Predict state using constant turn-rate equations
        x_k_k_minus_1 = [x1 + (2 * v * cos(h + (T * w) / 2) * sin((T * w) / 2)) / w;
                         x2 + (2 * v * sin(h + (T * w) / 2) * sin((T * w) / 2)) / w;
                         v;
                         h + T * w;
                         w];
        
        % State transition matrix for curved motion
        F_dot = [1, 0, (2*cos(h + (T*w)/2)*sin((T*w)/2))/w, -(2*v*sin(h + (T*w)/2)*sin((T*w)/2))/w, (T*v*cos(h + (T*w)/2)*cos((T*w)/2))/w - (T*v*sin(h + (T*w)/2)*sin((T*w)/2))/w - (2*v*cos(h + (T*w)/2)*sin((T*w)/2))/w^2;
                 0, 1, (2*sin(h + (T*w)/2)*sin((T*w)/2))/w,  (2*v*cos(h + (T*w)/2)*sin((T*w)/2))/w, (T*v*cos(h + (T*w)/2)*sin((T*w)/2))/w - (2*v*sin(h + (T*w)/2)*sin((T*w)/2))/w^2 + (T*v*sin(h + (T*w)/2)*cos((T*w)/2))/w;
                 0, 0,                                   1,                                      0,                                                                                           0;
                 0, 0,                                   0,                                      1,                                                                                           T;
                 0, 0,                                   0,                                      0,                                                                                           1];
    end

    % Process noise propagation matrix (Jacobian of noise model)
    G = [0.5 * T^2 * cos(h),       0;
         0.5 * T^2 * sin(h),       0;
                      T,           0;
                      0,   0.5 * T^2;
                      0,           T];

    % Process noise covariance for predicted state
    Q_k = G * Q * G';
end
