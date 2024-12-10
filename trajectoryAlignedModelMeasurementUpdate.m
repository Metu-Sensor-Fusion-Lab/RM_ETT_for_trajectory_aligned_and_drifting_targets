function [x_k_k, P_k_k, alpha_k_k, beta_k_k, EX_k_k] = trajectoryAlignedModelMeasurementUpdate(...
    x_k_k_minus_1, P_k_k_minus_1, alpha_k_k_minus_1, beta_k_k_minus_1, s, R, Y_k, I_max)
%TRAJECTORYALIGNEDMODELLMEASUREMENTUPDATE Performs the measurement update step 
% of a trajectory-aligned extended target tracking model using Variational Bayes.
%
% This function updates the posterior distributions of the target state and 
% extent parameters after receiving a batch of measurements. The approach 
% follows the derivations of a Variational Bayes-based extended target 
% tracking model for trajectory-aligned and drifting targets.
%
% Inputs:
%   x_k_k_minus_1      : Prior state estimate before receiving the current measurements
%   P_k_k_minus_1      : Prior state covariance before receiving the current measurements
%   alpha_k_k_minus_1  : Prior Inverse-Gamma shape parameters of the extent
%   beta_k_k_minus_1   : Prior Inverse-Gamma scale parameters of the extent
%   s                  : Scaling parameter used in the VB update equations
%   R                  : Measurement noise covariance matrix
%   Y_k                : Measurements at the current time step (2 x m_k)
%   I_max              : Maximum number of VB iterations for the measurement update
%
% Outputs:
%   x_k_k    : Posterior state estimate after measurement update
%   P_k_k    : Posterior state covariance after measurement update
%   alpha_k_k: Posterior Inverse-Gamma shape parameters of the extent
%   beta_k_k : Posterior Inverse-Gamma scale parameters of the extent
%   EX_k_k   : Posterior expectation of the extent matrix

    % Number of measurements at this scan
    m_k = size(Y_k, 2);

    % Initialize posterior distributions with prior values
    P_k_k = P_k_k_minus_1;                % State covariance initialization (Equation C1a)
    x_k_k = x_k_k_minus_1;                % State mean initialization (Equation C1b)
    alpha_k_k = alpha_k_k_minus_1;        % Extent shape parameter initialization (Equation C1c)
    beta_k_k = beta_k_k_minus_1;          % Extent scale parameter initialization (Equation C1d)

    % Initialize the measurement-related variables
    Sigma_z_k = diag(s*beta_k_k./(alpha_k_k-1)); % Measurement scatter matrix Sigma_z (Equation C1g)
    z_k = Y_k;                                  % Initialize "corrected" measurements z_k (Equation C1h)

    % Compute E[(sX)^{-1}] from current alpha_k_k, beta_k_k
    E_sX_1 = diag(alpha_k_k./(beta_k_k*s)); % Inverse-gamma expectation (Equation B3 in supplementary)

    % Define rotation and its derivative w.r.t angle, used in transformations
    T = @(theta) [cos(theta) -sin(theta); sin(theta) cos(theta)];    % Rotation matrix (Equation (7))
    T_dot = @(theta) [-sin(theta) -cos(theta); cos(theta) -sin(theta)]; % Derivative of rotation matrix (Equation (8))

    % T_switch_states is used to reorder state vector components for VB updates
    % Adjusting the partitioning between orientation and position/velocity states if needed.
    T_switch_states = [eye(3), zeros(3,2); zeros(2,3), (ones(2) - eye(2))];

    % H_phi extracts orientation-related states, H is measurement matrix for position
    H_phi = [0 0 0 1 0];
    H = [kron([1 0], eye(2)), zeros(2,1)]; % Measurement matrix extracting x,y from state (no direct angle measurement)

    %% Variational Bayes Iterations
    for l = 1:I_max
        % At each iteration, we re-parameterize and compute expectations to 
        % update q_x, q_Z, q_Gamma distributions.

        %% Re-parametrization (Equation B8 and related steps)
        % Switch states arrangement to handle orientation and linear states properly
        x_k_k = T_switch_states*x_k_k;
        P_k_k = T_switch_states*P_k_k*T_switch_states.';
        H_phi = H_phi*T_switch_states;
        H_x = H(:, 1:end-1);

        % Extract relevant terms
        phi_hat = H_phi*x_k_k;          % Orientation-related mean
        sigma_phi = H_phi*P_k_k*H_phi.';% Orientation-related variance
        x_x = x_k_k(1:end-1);           % Extract linear state components
        P_x = P_k_k(1:end-1, 1:end-1);  % Extract linear state covariance
        P_xphi = P_k_k(1:end-1, end);   % Cross-covariance between linear states and orientation

        %% Update of q_Gamma (Equation 30a and 30b)
        % Update the Inverse-Gamma parameters for the extent
        alpha_k_k = alpha_k_k_minus_1 + m_k/2; % Equation (30a) - update shape parameter

        E_xx_zz_zx_xz = zeros(2, 2);
        % Compute expected values involving z, x, and transformations
        for j=1:m_k
            % E_zz and E_xx_xz_zx terms computed via helper expectations:
            E_zz = eTMT(phi_hat, sigma_phi, (z_k(:,j)*z_k(:,j).' + Sigma_z_k), 2);
            E_xz = eTMT(phi_hat, sigma_phi, H_x*(x_x - (P_xphi/sigma_phi)*phi_hat)*z_k(:, j).', 2) ...
                + eThetaTMT(phi_hat, sigma_phi, H_x*(P_xphi/sigma_phi)*z_k(:, j).', 2);
            E_xx = ...
                eTMT(phi_hat, sigma_phi, H_x*(P_x + x_x*x_x.'+ ((phi_hat/sigma_phi)^2 - 1/sigma_phi)*(P_xphi*P_xphi.') - (phi_hat/sigma_phi)*(x_x*P_xphi.' + P_xphi*x_x.'))*H_x.',  2) ...
                + eThetaTMT(phi_hat, sigma_phi, H_x*((1/sigma_phi)*(x_x*P_xphi.' + P_xphi*x_x.') - 2*(phi_hat/(sigma_phi^2))*(P_xphi*P_xphi.'))*H_x.', 2) ...
                + eTheta2TMT(phi_hat, sigma_phi, H_x*((P_xphi*P_xphi.')/(sigma_phi^2))*H_x.', 2);

            % Summation of terms to form the combined expectation needed for beta_k_k update
            E_xx_zz_zx_xz = E_xx_zz_zx_xz + E_zz + E_xx - E_xz - E_xz.';
        end

        % Update scale parameter of Inverse-Gamma distribution (Equation 30b)
        beta_k_k = beta_k_k_minus_1 + diag(E_xx_zz_zx_xz)/(2*s);

        %% Update q_Z (Equation 33a, 33b)
        % Update the measurement distribution parameters
        Sigma_z_k = (R^-1 + eTMT(phi_hat, sigma_phi, E_sX_1, 1))^-1; % (33a)
        for j=1:m_k
            z_k(:, j) = Sigma_z_k*( R\Y_k(:, j) ...
                + eTMT(phi_hat, sigma_phi, E_sX_1, 1)*H*[x_x - (P_xphi/sigma_phi)*phi_hat; 0] ...
                + eThetaTMT(phi_hat, sigma_phi, E_sX_1, 1)*H*[(P_xphi/sigma_phi); 1] );  % (33b)
        end

        % Revert the state switching
        x_k_k = T_switch_states*x_k_k;
        P_k_k = T_switch_states*P_k_k*T_switch_states.';
        H_phi = H_phi*T_switch_states;

        %% Update q_x (Equation 27a, 27b)
        % Update the state posterior distribution
        % Compute necessary terms (PHI, phi) for q_x update
        phi = zeros(size(x_k_k));
        PHI = zeros(size(P_k_k));

        for j = 1:m_k
            PHI = PHI ...
                + H_phi.'*trace((Sigma_z_k + (z_k(:, j) - H*x_k_k)*(z_k(:, j) - H*x_k_k).') ...
                *T_dot(H_phi*x_k_k)*E_sX_1*T_dot(H_phi*x_k_k).')*H_phi ...
                - H.'*T(H_phi*x_k_k)*E_sX_1*T_dot(H_phi*x_k_k).'*(z_k(:, j) - H*x_k_k)*H_phi ...
                - (H.'*T(H_phi*x_k_k)*E_sX_1*T_dot(H_phi*x_k_k).'*(z_k(:, j) - H*x_k_k)*H_phi).' ...
                + H.'*T(H_phi*x_k_k)*E_sX_1*T(H_phi*x_k_k).'*H; % Combined (27c) by (B5)

            phi = phi ...
                + H_phi.'*(trace((Sigma_z_k + (z_k(:, j) - H*x_k_k)*(z_k(:, j) - H*x_k_k).') ...
                *T_dot(H_phi*x_k_k)*E_sX_1*(T_dot(H_phi*x_k_k).'*(H_phi*x_k_k) - T(H_phi*x_k_k).')) ...
                - (z_k(:, j) - H*x_k_k).'*T_dot(H_phi*x_k_k)*E_sX_1*T(H_phi*x_k_k).'*H*x_k_k) ...
                + H.'*T(H_phi*x_k_k)*E_sX_1*(T(H_phi*x_k_k).'*(z_k(:, j) - H*x_k_k) ...
                - T_dot(H_phi*x_k_k).'*(z_k(:, j) - H*x_k_k)*H_phi*x_k_k + T(H_phi*x_k_k).'*H*x_k_k); % (27d) by (B6)
        end

        % State covariance and mean update (27a, 27b)
        P_k_k = (inv(P_k_k_minus_1) + PHI)^-1; 
        x_k_k = P_k_k*(P_k_k_minus_1\x_k_k_minus_1 + phi); 

    end

    %% Compute the final expected extent matrix EX_k_k
    % Use final states and distribution parameters
    EX_k_k =  eTMT(H_phi*x_k_k, H_phi*P_k_k*H_phi.', diag(beta_k_k./(alpha_k_k-1)), 1);

end


%% Helper Expectation Functions
% The following helper functions compute expectations of transformed matrices 
% under random rotations characterized by a random angle theta and its distribution.

function expectation = eTMT(theta_hat, sigma_theta, M, form)
% eTMT:
%   Computes E[T*M*T^T], where T is a rotation matrix dependent on a 
%   random angle with mean theta_hat and variance sigma_theta.
%
% Inputs:
%   theta_hat    : mean angle
%   sigma_theta  : angle variance
%   M            : matrix to be transformed
%   form         : a flag indicating whether to use a certain form (1 or 2)
%
% Output:
%   expectation  : The expected transformed matrix under T and angle distribution.

    % Extract elements of M for vectorization
    m11 = M(1,1);
    m21 = M(2,1);
    m12 = M(1,2);
    m22 = M(2,2);

    th = theta_hat;
    ts = sigma_theta;

    % Reshape M into a form used by given equations (reference B in paper)
    vec_M = [m11,  m22, -(m12 + m21);
             m21, -m12,  (m11 - m22);
             m12, -m21,  (m11 - m22);
             m22,  m11,  (m12 + m21)];

    if form ~= 1
        % Adjust signs depending on form selection
        vec_M(:, end) = -vec_M(:, end);
    end

    % Compute vector with angle-dependent terms: cos(2th), sin(2th), and exp(-2ts)
    t_vec = [1+cos(2*th)*exp(-2*ts), 1 - cos(2*th)*exp(-2*ts), sin(2*th)*exp(-2*ts)].';

    % Calculate expectation
    expectation = reshape(vec_M*t_vec, 2, 2)*0.5;
end

function expectation = eThetaTMT(theta_hat, sigma_theta, M, form)
% eThetaTMT:
%   Computes E[theta*T*M*T^T], an expectation involving the angle and its rotations.
%
% Inputs:
%   theta_hat   : mean angle
%   sigma_theta : angle variance
%   M           : matrix to be transformed
%   form        : form selection for sign adjustments
%
% Output:
%   expectation : The expected matrix for the given expression

    % Extract elements of M
    m11 = M(1,1);
    m21 = M(2,1);
    m12 = M(1,2);
    m22 = M(2,2);

    th = theta_hat;
    ts = sigma_theta;

    % Vectorization of M as in eTMT
    vec_M = [m11,  m22, -(m12 + m21);
             m21, -m12,  (m11 - m22);
             m12, -m21,  (m11 - m22);
             m22,  m11,  (m12 + m21)];

    if form ~= 1
        vec_M(:, end) = -vec_M(:, end);
    end

    % Matrix encoding the first and second order moments of theta
    t_mat = [th,   th,   -2*ts;
             th,  -th,    2*ts;
             0,   2*ts,   th ];

    % Vector with sin/cos/exp terms as before, slightly different order here
    t_vec = [1, exp(-2*ts)*cos(2*th), exp(-2*ts)*sin(2*th)].';

    % Compute expectation
    expectation = reshape(vec_M*t_mat*t_vec, 2, 2)*0.5;

end

function expectation = eTheta2TMT(theta_hat, sigma_theta, M, form)
% eTheta2TMT:
%   Computes E[theta^2*T*M*T^T], involving second-order angle terms.
%
% Inputs:
%   theta_hat   : mean angle
%   sigma_theta : angle variance
%   M           : matrix to be transformed
%   form        : form selection for sign adjustments
%
% Output:
%   expectation : The expected matrix considering second-moment terms of theta

    % Extract M elements
    m11 = M(1,1);
    m21 = M(2,1);
    m12 = M(1,2);
    m22 = M(2,2);

    th = theta_hat;
    ts = sigma_theta;

    % Vectorization as before
    vec_M = [m11,  m22, -(m12 + m21);
             m21, -m12,  (m11 - m22);
             m12, -m21,  (m11 - m22);
             m22,  m11,  (m12 + m21)];

    if form ~= 1
        vec_M(:, end) = -vec_M(:, end);
    end

    % Matrix for second order moments including theta^2 and ts terms
    % Derived from expansions involving angle distributions
    t_mat = [ts + th^2,   ts + th^2 - 4*ts^2,       -4*ts*th;
             ts + th^2,  -ts - th^2 + 4*ts^2,       4*ts*th;
             0,           4*ts*th,   ts+th^2-4*ts^2];

    % Vector with angle-dependent terms
    t_vec = [1, exp(-2*ts)*cos(2*th), exp(-2*ts)*sin(2*th)].';

    expectation = reshape(vec_M*t_mat*t_vec, 2, 2)*0.5;

end
