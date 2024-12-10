function [x_k_k, P_k_k, alpha_k_k, beta_k_k, theta_k_k, Theta_k_k, EX_k_k] =  driftModelMeasurementUpdate(...
    x_k_k_minus_1, P_k_k_minus_1, alpha_k_k_minus_1, beta_k_k_minus_1, ...
    theta_k_k_minus_1, Theta_k_k_minus_1, s, R, Y_k, I_max)
%DRIFTMODELMEASUREMENTUPDATE Performs the measurement update for a drift model in extended target tracking.
%
% This function updates the posterior distributions of the target state, orientation, and extent
% parameters given a set of measurements. It uses a variational Bayes approach to update the 
% distributions iteratively.
%
% Inputs:
%   x_k_k_minus_1       : Prior state estimate vector before current measurement update
%   P_k_k_minus_1       : Prior state covariance matrix before current measurement update
%   alpha_k_k_minus_1   : Prior Inverse-Gamma shape parameters of the extent
%   beta_k_k_minus_1    : Prior Inverse-Gamma scale parameters of the extent
%   theta_k_k_minus_1   : Prior orientation parameters (e.g. [theta; theta_dot]) before update
%   Theta_k_k_minus_1    : Prior orientation covariance (e.g. 2x2) before update
%   s                   : Scaling parameter used in VB update equations
%   R                   : Measurement noise covariance matrix
%   Y_k                 : Current set of measurements (2 x m_k)
%   I_max               : Maximum number of VB iterations for the measurement update
%
% Outputs:
%   x_k_k       : Posterior state estimate after measurement update
%   P_k_k       : Posterior state covariance after measurement update
%   alpha_k_k   : Posterior Inverse-Gamma shape parameters of the extent
%   beta_k_k    : Posterior Inverse-Gamma scale parameters of the extent
%   theta_k_k    : Posterior orientation parameters after measurement update
%   Theta_k_k    : Posterior orientation covariance after measurement update
%   EX_k_k       : Posterior expected extent matrix after measurement update

    % Dimension of the extent (2D)
    d = length(alpha_k_k_minus_1);

    % Measurement matrix H assumed as [I_2x2 0_2x2]
    % If different model is used, adjust H accordingly.
    H = [eye(2), zeros(2,2)];

    % Initialize the posteriors with their priors
    x_k_k = x_k_k_minus_1;
    P_k_k = P_k_k_minus_1;
    alpha_k_k = alpha_k_k_minus_1;
    beta_k_k = beta_k_k_minus_1;
    theta_k_k = theta_k_k_minus_1;
    Theta_k_k = Theta_k_k_minus_1;

    % Number of measurements at current time step
    m_k = size(Y_k, 2);

    % Initial guessed measurement scatter and corrected measurements
    Sigma_k = diag(s*beta_k_k./(alpha_k_k-1));
    z_k = Y_k;

    % Vector TH used for extracting angle-related parameters (e.g. TH*theta_k_k = theta)
    TH = [1 0];

    % Iterative VB Update
    % Perform I_max iterations to refine the posterior distributions
    for l = 1:I_max
        % Compute E[(sX)^{-1}] from alpha_k_k and beta_k_k
        E_sX_1 = diag(alpha_k_k./(beta_k_k*s));

        % Compute E[T*M*T^T] for T related to orientation (theta)
        % E_T_sGammaInv_TT is E[T*(sX)^{-1}*T^T] used in state update steps
        E_T_sGammaInv_TT = eTMT(TH*theta_k_k, TH*Theta_k_k*TH.' , E_sX_1,1);

        

        % Update q_x (State distribution)
        % Update state covariance (40a) and mean (40b)
        P_k_k = (P_k_k_minus_1^-1 + m_k*H'*E_T_sGammaInv_TT*H)^-1;
        x_k_k = P_k_k*(P_k_k_minus_1\x_k_k_minus_1 + m_k*H'*E_T_sGammaInv_TT*mean(z_k,2));

        % Update q_Z (Measurement distribution)
        % Update Sigma_k and z_k based on new state and extent estimates (46a & 46b)
        Sigma_k = (E_T_sGammaInv_TT + R^-1)^-1;
        for j=1:m_k
            z_k(:,j) = Sigma_k*(E_T_sGammaInv_TT*H*x_k_k + R\Y_k(:,j));
        end

        % Compute expected measurement error terms E_zzT for each measurement
        for j=1:m_k
            E_zzT(:,:,j) = H*P_k_k*H' + Sigma_k + (z_k(:,j) - H*x_k_k)*(z_k(:,j) - H*x_k_k)';
            E_TT_zz_T(:,:,j) = eTMT(TH*theta_k_k, TH*Theta_k_k*TH.', E_zzT(:,:,j), 2);
        end
        % Update q_X (Extent parameters)
        % Update alpha_k_k and beta_k_k (21a & 21b)
        alpha_k_k = alpha_k_k_minus_1 + m_k*0.5;
        for ii=1:d
            E_zz_ii = 0;
            for j=1:m_k
                E_zz_ii = E_zz_ii + E_TT_zz_T(ii,ii,j);
            end
            beta_k_k(ii) = beta_k_k_minus_1(ii) + E_zz_ii/(2*s);
        end

        % Update q_theta (Orientation parameters)
        % Orientation update (51a & 51b, analogous to equations related to q_theta)
        delta = 0; Delta = 0;
        T_dot_theta = rotationMatrixD(TH*theta_k_k);
        T_theta = rotationMatrix(TH*theta_k_k);
        E_sX_1 = diag(alpha_k_k./(beta_k_k*s));

        for j = 1:m_k
            % Delta and delta accumulate orientation-related terms
            Delta = Delta + TH'*trace(E_sX_1*T_dot_theta'*E_zzT(:,:,j)*T_dot_theta)*TH; 
            delta = delta + TH'*trace(E_sX_1*T_dot_theta'*E_zzT(:,:,j)*T_dot_theta)*TH*theta_k_k ...
                - TH'*trace(E_sX_1*T_theta'*E_zzT(:,:,j)*T_dot_theta); 
        end
        Theta_k_k = (Theta_k_k_minus_1^-1 + Delta)^-1;
        theta_k_k = Theta_k_k*(Theta_k_k_minus_1\theta_k_k_minus_1 + delta);

    end

    % Compute final expected extent matrix EX_k_k from updated parameters
    EX_k_k = eTMT(TH*theta_k_k, TH*Theta_k_k*TH.', diag(beta_k_k./(alpha_k_k-1)), 1);

end

%% Helper Functions

function expectation = eTMT(theta_hat, sigma_theta, M, form)
% eTMT calculates E[T*M*T^T], where T is a rotation matrix dependent on a
% normally distributed angle with mean theta_hat and variance sigma_theta.
%
% Inputs:
%   theta_hat   : mean angle
%   sigma_theta : angle variance (scalar)
%   M           : 2x2 matrix to be transformed
%   form        : a mode indicator (1 or 2) affecting the sign pattern
%
% Output:
%   expectation : 2x2 expected transformed matrix

    m11 = M(1,1);
    m21 = M(2,1);
    m12 = M(1,2);
    m22 = M(2,2);

    th = theta_hat;
    ts = sigma_theta;

    % Arrange M into a vectorized form used by the expansions in the VB derivation
    vec_M = [m11,  m22, -(m12 + m21);
             m21, -m12,  (m11 - m22);
             m12, -m21,  (m11 - m22);
             m22,  m11,  (m12 + m21)];

    if form ~= 1
        % Adjust signs based on form
        vec_M(:, end) = -vec_M(:, end);
    end

    % Construct vector with angle-dependent terms: involves cos(2theta), sin(2theta) and exp(-2ts)
    t_vec = [1+cos(2*th)*exp(-2*ts), 1 - cos(2*th)*exp(-2*ts), sin(2*th)*exp(-2*ts)].';

    % Final expectation
    expectation = reshape(vec_M*t_vec, 2, 2)*0.5;
end

function R = rotationMatrix(angle)
% rotationMatrix returns a 2D rotation matrix for the given angle
    R = [cos(angle), -sin(angle);
         sin(angle),  cos(angle)];
end

function Rd = rotationMatrixD(angle)
% rotationMatrixD returns the derivative of the rotation matrix wrt the angle
% This represents the rate of change of the rotation matrix with respect to the angle.
    Rd = [-sin(angle), -cos(angle);
          cos(angle),  -sin(angle)];
end
