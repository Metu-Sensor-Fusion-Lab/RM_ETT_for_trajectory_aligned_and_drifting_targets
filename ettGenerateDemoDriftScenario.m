function [ett_measurements, ett_ground_truth] = ettGenerateDemoDriftScenario( ...
    t, ... % simulation time vector
    state, ... % initial state
    extent, ... % initial extent
    H, ... % measurement matrix
    R, ... % measurement covariance
    mean_num_of_meas)

%ETTGENERATESCENARIO Summary of this function goes here
%   Detailed explanation goes here
turn_angles = [0, pi, pi, 2*pi/3, 0];
turn_fractions = [.9, .9, .9, .9, .9];
section_durations = [.1, .3, .2, .3, .1];

sections = size(turn_fractions, 2);
size_t = size(t,2);
ett_ground_truth.states = double.empty(size(state, 1), 0);
ett_ground_truth.extents = double.empty(size(extent, 1), size(extent, 2), 0);
ett_measurements = {};

last_index = 1;

for i=1:sections-1

    t_section = t(last_index : fix(size_t*section_durations(i)) + last_index + 1);
    [section_measurements, section_ground_truth] = ettGenerateDrift(t_section, state, extent, H, R, mean_num_of_meas, turn_fractions(i), turn_angles(i));
    state = section_ground_truth.states(:, end);
    extent = section_ground_truth.extents(:, :, end);
    section_ground_truth.states(:, end) = [];
    section_ground_truth.extents(:, :, end) = [];
    section_measurements(end) = [];

    ett_ground_truth.states = [ett_ground_truth.states section_ground_truth.states];
    ett_ground_truth.extents = cat(3, ett_ground_truth.extents, section_ground_truth.extents);
    ett_measurements = [ett_measurements section_measurements];

    last_index = last_index + size(t_section, 2) - 1;


end

t_section = t(last_index : fix(size_t*section_durations(i+1)) + last_index - sections);
[section_measurements, section_ground_truth] = ettGenerateDrift(t_section, state, extent, H, R, mean_num_of_meas, turn_fractions(i+1), turn_angles(i+1));

ett_ground_truth.states = [ett_ground_truth.states(:,1), ett_ground_truth.states, section_ground_truth.states];
ett_ground_truth.extents = cat(3, ett_ground_truth.extents(:,:,1), ett_ground_truth.extents, section_ground_truth.extents);
ett_measurements = [ett_measurements section_measurements];


end

function [ett_measurements, ett_ground_truth] = ettGenerateDrift( ...
    t, ... % simulation time vector
    state, ... % initial state
    extent, ... % initial extent
    H, ... % measurement matrix
    R, ... % measurement covariance
    mean_num_of_meas, ...
    turn_frac, ... % fraction of time should be spent on turning
    turn_angle ... % half turn quarter turn etc
)
%ETT_GENERATE_DRIFT Generates A drifting scenario

% Create a continuous U turn with 3 segments
% Create a continuous U turn with 3 segments
num_of_frames = size(t,2);
T = (t(end) - t(1))/(num_of_frames - 1);
states = zeros(6, num_of_frames); % x y vx vy theta omega
extents = zeros(2,2, num_of_frames);
heading_angle = state(5); % in radians
turn_rate = turn_angle/((t(end) - t(1))*turn_frac); % rad/s

states(:, 1) = [state(1:4); heading_angle; 0];
extents(:,:,1) = extent;

for k = 2:num_of_frames
    
    % first segment, constant velocity through positive x direction
    if k < num_of_frames*(1 - turn_frac)/2
        states(:, k) = states(:, k-1);
        states(1:2, k) = states(1:2, k) + states(3:4, k)*T;
        extents(:,:,k) = extents(:,:,k-1);
    
    % second segment, constant turn rate model
    elseif k >= num_of_frames*(1 - turn_frac)/2 && k < num_of_frames*(1 + turn_frac/2)/2
        states(3:4, k) = rotationMatrix(turn_rate*T)*states(3:4, k-1);
        states(1:2, k) = states(1:2, k-1) + states(3:4, k-1)*T;
        heading_angle = heading_angle +  1.5*turn_rate*T;
        states(5, k) = heading_angle;
        states(6, k) = turn_rate;
        extents(:,:,k) = rotationMatrix(1.5*turn_rate*T)*extents(:,:,k-1)*rotationMatrix(1.5*turn_rate*T)';
        
    elseif k >= num_of_frames*(1 + turn_frac/2)/2 && k < num_of_frames*(1 + turn_frac)/2
        states(3:4, k) = rotationMatrix(turn_rate*T)*states(3:4, k-1);
        states(1:2, k) = states(1:2, k-1) + states(3:4, k-1)*T;
        states(5, k) = states(5, k-1);
        states(6, k) = 0;
        extents(:,:,k) = extents(:,:,k-1);
        
    % third segment, constant velocity through negative x direction
    elseif k >= num_of_frames*(1 + turn_frac)/2
        states(:, k) = states(:, k-1);
        states(1:2, k) = states(1:2, k) + states(3:4, k)*T;
        states(5, k) = states(5, k-1);
        states(6, k) = 0;
        extents(:,:,k) = extents(:,:,k-1);      
    end
    
end

Y_k = cell(1,k);
for k = 1:num_of_frames
    
    state = states(:,k);
    extent = extents(:,:,k);
    
    m_k = poissrnd(mean_num_of_meas);
    while m_k == 0
        m_k = poissrnd(mean_num_of_meas); % failsafe
    end
    
    [V,D] = eig(extent);
    Y_k{k} = mvnrnd(H*state(1:4), .25*extent + R, m_k)';
    
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
