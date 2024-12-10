function [ett_measurements, ett_ground_truth] = ettGenerateDemoAlignedScenario(t, state, extent, H, R, mean_num_of_meas)
% ETTGENERATEDEMOALIGNEDSCENARIO:
% Generates a synthetic scenario for a trajectory-aligned extended target
% with turning motion. The scenario consists of multiple segments, some with
% straight-line motion and others with turning motion.
%
% Inputs:
%   t                : Simulation time vector
%   state            : Initial state vector [x; y; vx; vy; theta; omega]
%   extent           : Initial extent matrix (2D ellipse)
%   H                : Measurement matrix
%   R                : Measurement covariance matrix
%   mean_num_of_meas : Average number of measurements per scan
%
% Outputs:
%   ett_measurements : Cell array containing measurements for each time step
%   ett_ground_truth : Struct containing ground truth states and extents

    % Predefined parameters for the turning motion
    turn_angles = [pi, -pi, -3*pi/2];         % Turning angles for each segment
    turn_fractions = [0.35, 0.8, 0.65];      % Fraction of time spent turning in each segment
    section_durations = [0.35, 0.25, 0.4];   % Relative duration of each section

    % Initialize variables
    sections = size(turn_fractions, 2);      % Number of motion segments
    size_t = size(t, 2);                     % Total number of time steps
    ett_ground_truth.states = double.empty(size(state, 1), 0); % Ground truth states
    ett_ground_truth.extents = double.empty(size(extent, 1), size(extent, 2), 0); % Ground truth extents
    ett_measurements = {};                   % Store generated measurements

    last_index = 1; % Index for the start of each section

    % Generate motion segments except the last one
    for i = 1:sections-1
        % Extract time steps for this segment
        t_section = t(last_index : fix(size_t * section_durations(i)) + last_index + 1);

        % Generate U-turn motion and measurements for this section
        [section_measurements, section_ground_truth] = ettGenerateUTurn(...
            t_section, state, extent, H, R, mean_num_of_meas, turn_fractions(i), turn_angles(i));

        % Update the state and extent for the next segment
        state = section_ground_truth.states(:, end);   % Final state of this segment
        extent = section_ground_truth.extents(:, :, end); % Final extent of this segment

        % Remove the last element for seamless concatenation
        section_ground_truth.states(:, end) = [];
        section_ground_truth.extents(:, :, end) = [];
        section_measurements(end) = [];

        % Concatenate results
        ett_ground_truth.states = [ett_ground_truth.states, section_ground_truth.states];
        ett_ground_truth.extents = cat(3, ett_ground_truth.extents, section_ground_truth.extents);
        ett_measurements = [ett_measurements, section_measurements];

        % Update index for the next section
        last_index = last_index + size(t_section, 2) - 1;
    end

    % Generate the final motion segment
    t_section = t(last_index : fix(size_t * section_durations(end)) + last_index - sections);
    [section_measurements, section_ground_truth] = ettGenerateUTurn(...
        t_section, state, extent, H, R, mean_num_of_meas, turn_fractions(end), turn_angles(end));

    % Concatenate the last section and add initial state to the start
    ett_ground_truth.states = [ett_ground_truth.states(:, 1), ett_ground_truth.states, section_ground_truth.states];
    ett_ground_truth.extents = cat(3, ett_ground_truth.extents(:, :, 1), ett_ground_truth.extents, section_ground_truth.extents);
    ett_measurements = [ett_measurements, section_measurements];
end

function [ett_measurements, ett_ground_truth] = ettGenerateUTurn(t, state, extent, H, R, mean_num_of_meas, turn_frac, turn_angle)
% ETTGENERATEUTURN:
% Generates a U-turn trajectory for an extended target and simulates measurements.
%
% Inputs:
%   t                : Simulation time vector
%   state            : Initial state vector [x; y; vx; vy; theta; omega]
%   extent           : Initial extent matrix (2D ellipse)
%   H                : Measurement matrix
%   R                : Measurement covariance matrix
%   mean_num_of_meas : Average number of measurements per scan
%   turn_frac        : Fraction of time spent turning
%   turn_angle       : Total angle turned during the segment (radians)
%
% Outputs:
%   ett_measurements : Cell array containing measurements for this segment
%   ett_ground_truth : Struct containing ground truth states and extents for this segment

    num_of_frames = size(t, 2);         % Number of time steps in this segment
    T = (t(end) - t(1)) / (num_of_frames - 1); % Time step duration
    states = zeros(6, num_of_frames);  % Ground truth states [x, y, vx, vy, theta, omega]
    extents = zeros(2, 2, num_of_frames); % Ground truth extents

    % Initial conditions
    heading_angle = state(5); % Initial heading angle
    turn_rate = turn_angle / ((t(end) - t(1)) * turn_frac); % Turn rate (rad/s)
    states(:, 1) = [state(1:4); heading_angle; 0]; % Set initial state
    extents(:, :, 1) = extent; % Set initial extent

    % Generate ground truth trajectory
    for k = 2:num_of_frames
        if k < num_of_frames * (1 - turn_frac) / 2
            % Segment 1: Straight-line motion
            states(:, k) = states(:, k-1);
            states(1:2, k) = states(1:2, k) + states(3:4, k) * T;
            extents(:, :, k) = extents(:, :, k-1);

        elseif k >= num_of_frames * (1 - turn_frac) / 2 && k <= num_of_frames * (1 + turn_frac) / 2
            % Segment 2: Turning motion
            states(3:4, k) = rotationMatrix(turn_rate * T) * states(3:4, k-1); % Rotate velocity
            states(1:2, k) = states(1:2, k-1) + rotationMatrix(turn_rate * T / 2) * states(3:4, k-1) * T; % Update position
            heading_angle = heading_angle + turn_rate * T; % Update heading angle
            states(5, k) = heading_angle; % Update heading angle in state
            states(6, k) = turn_rate; % Set turn rate in state
            [V, D] = eig(extents(:, :, k-1));
            V = rotationMatrix(turn_rate * T) * V; % Rotate extent
            extents(:, :, k) = V * D * V';

        else
            % Segment 3: Straight-line motion
            states(:, k) = states(:, k-1);
            states(1:2, k) = states(1:2, k) + states(3:4, k) * T;
            states(6, k) = 0; % Set turn rate to zero
            extents(:, :, k) = extents(:, :, k-1);
        end
    end

    % Generate measurements based on ground truth
    Y_k = cell(1, num_of_frames);
    for k = 1:num_of_frames
        state = states(:, k);
        extent = extents(:, :, k);
        m_k = poissrnd(mean_num_of_meas); % Number of measurements
        while m_k == 0
            m_k = poissrnd(mean_num_of_meas); % Retry if zero measurements
        end

        [V, D] = eig(extent); % Decompose extent for measurement generation
        variances = diag(sqrt(D));
        rand_rectangle = [-variances(1) + 2 * variances(1) * rand(1, m_k); ...
                          -variances(2) + 2 * variances(2) * rand(1, m_k)];
        in_ellipse = rand_rectangle(1, :).^2 / (variances(1))^2 + rand_rectangle(2, :).^2 / (variances(2))^2 < 1;
        rand_ellipse = rand_rectangle(:, in_ellipse);

        Y_k{k} = mvnrnd([0, 0], R, size(rand_ellipse, 2))' + V * rand_ellipse + H * state(1:4);
    end

    ett_measurements = Y_k;
    ett_ground_truth.states = states;
    ett_ground_truth.extents = extents;
end

function T = rotationMatrix(theta)
% ROTATIONMATRIX:
% Computes the 2D rotation matrix for a given angle.
%
% Input:
%   theta : Angle in radians
%
% Output:
%   T     : 2D rotation matrix

    T = [cos(theta), -sin(theta); ...
         sin(theta),  cos(theta)];
end
