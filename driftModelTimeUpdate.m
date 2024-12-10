function [x_predicted, P_predicted, alpha_predicted, beta_predicted, theta_predicted, Theta_predicted] = driftModelTimeUpdate(...
    x_k_k, P_k_k, alpha_k_k, beta_k_k, Q, theta_k_k, Theta_k_k, TQ, T, tau)
%DRIFTMODELTIMEUPDATE Performs the time (prediction) update step for a drift model in extended target tracking.
%
% This function predicts the target state, orientation, and extent parameters forward 
% one time step. It applies a constant-velocity model for the linear state 
% components and a separate orientation prediction model for the angle-related 
% states. The Inverse-Gamma parameters for the extent distribution are exponentially 
% decayed to implement a forgetting factor as described in related VB-based 
% extended target tracking approaches.
%
% Inputs:
%   x_k_k         : Posterior state estimate at the current time step
%   P_k_k         : Posterior state covariance at the current time step
%   alpha_k_k     : Posterior Inverse-Gamma shape parameters at current step
%   beta_k_k      : Posterior Inverse-Gamma scale parameters at current step
%   Q             : Process noise covariance for the linear (position/velocity) state
%   theta_k_k     : Posterior orientation parameters (e.g. [theta; theta_dot]) at current step
%   Theta_k_k      : Posterior orientation covariance at current step
%   TQ            : Process noise parameter for the orientation model
%   T             : Time step duration
%   tau           : Time constant for exponential forgetting factor
%
% Outputs:
%   x_predicted    : Predicted state for the next time step
%   P_predicted    : Predicted state covariance for the next time step
%   alpha_predicted: Predicted Inverse-Gamma shape parameters for the extent
%   beta_predicted : Predicted Inverse-Gamma scale parameters for the extent
%   theta_predicted: Predicted orientation parameters for the next time step
%   Theta_predicted: Predicted orientation covariance for the next time step

    % Predict the linear state and covariance (Equations (D1a) and (D1b))
    [x_predicted, F_dot_k, Q_k] = modelConstantVelocity(x_k_k, Q, T);
    P_predicted = F_dot_k*P_k_k*F_dot_k' + Q_k;

    % Predict the orientation state and covariance (Equations (D1c) and (D1d))
    [theta_predicted, TFdot, TQ_k] = thetaPrediction(theta_k_k, TQ, T);
    Theta_predicted = TFdot*Theta_k_k*TFdot' + TQ_k;

    % Exponentially decay the Inverse-Gamma parameters for the extent (Equations (D1e) and (D1f))
    alpha_predicted = alpha_k_k*exp(-T/tau);
    beta_predicted = beta_k_k*exp(-T/tau);

end

function [x_k_k_minus_1, F_dot, Q_k] = modelConstantVelocity(x, Q, T)
%MODELCONSTANTVELOCITY Propagates the state and covariance assuming a constant-velocity model.
%
% The state is assumed to be [x; y; vx; vy]. The model predicts:
% x_{k+1} = x_k + vx_k*T
% y_{k+1} = y_k + vy_k*T
% vx_{k+1} = vx_k
% vy_{k+1} = vy_k
%
% Inputs:
%   x : Current state [x; y; vx; vy]
%   Q : Process noise covariance for linear motion
%   T : Time step
%
% Outputs:
%   x_k_k_minus_1 : Predicted state after time T
%   F_dot          : State transition Jacobian
%   Q_k            : Discretized process noise covariance

    % State prediction
    x_k_k_minus_1 = [ x(1) + T*x(3);
                      x(2) + T*x(4);
                      x(3);
                      x(4) ];

    % State transition matrix (Jacobian)
    % For constant velocity in 2D:
    % [1 0 T 0
    %  0 1 0 T
    %  0 0 1 0
    %  0 0 0 1]
    % Using Kronecker product to construct this compactly:
    F_dot = kron([1, T; 0, 1], eye(2));

    % Process noise covariance for discretized motion:
    % G = integral over [0,T] of exp(Ft)BW W^T(exp(Ft))^T dt simplified leads to:
    G = [T^3/3, T^2/2; T^2/2, T];
    Q_k = kron(G, Q);
end

function [theta_k_k_minus_1, TFdot, TQ_k] = thetaPrediction(theta_k_k, Q, T)
%THETAPREDICTION Predicts the orientation state forward one time step.
%
% Here, theta_k_k is assumed to be [theta; theta_dot].
% Model:
% theta_{k+1} = theta_k + theta_dot_k*T
% theta_dot_{k+1} = theta_dot_k
%
% Inputs:
%   theta_k_k : Current orientation state [theta; theta_dot]
%   Q         : Process noise parameter for orientation
%   T         : Time step
%
% Outputs:
%   theta_k_k_minus_1 : Predicted orientation state
%   TFdot             : State transition matrix for orientation
%   TQ_k              : Orientation process noise covariance

    t = theta_k_k;

    % State prediction
    theta_k_k_minus_1 = [t(1) + T*t(2);
                         t(2)];

    % Orientation state transition matrix
    TFdot = [1 T;
             0 1];

    % Process noise for orientation:
    % G = [T^2/2; T] and Q is scalar process noise intensity
    G = [T^2/2; T];
    TQ_k = G*Q*G.';
end
