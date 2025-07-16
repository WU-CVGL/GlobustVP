function [gt_correspond_list, all_2D_lines_norm, uncertainty, parallel_line_list, MF_GT, K] = Synthetic_GenerateData(outlierRatio, param)

    %% Step one: generate all the inliers
    num_1 = param.line_seg(1); % ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    num_2 = param.line_seg(2);
    num_3 = param.line_seg(3);
    totalNum = num_1 + num_2 + num_3;
    
    K = [800, 0, 320;
        0, 800, 240;
        0, 0, 1];
    
    gt_correspond_list = [];
    
    % 3D lines
    [first_3D_lines, second_3D_lines, third_3D_lines, MF_GT] = generate3Dlines(num_1, num_2, num_3, totalNum, param, K);
    
    id_point = 0;
    for point_i = 1:param.vanishing_point_num
        point_single = zeros(1, totalNum);
        point_single(id_point + 1:id_point + param.line_seg(point_i)) = 1;
        gt_correspond_list = [gt_correspond_list; point_single];
        id_point = id_point + param.line_seg(point_i);
    end
    
    % 2D lines
    Rw2c = eye(3,3);
    tw2c = [0;0;0];
        
    rng('shuffle');
    sigma = param.noise;
    noise_1_s_x = sigma*rand(1, num_1)-0.5*sigma*ones(1, num_1);
    noise_1_s_y = sigma*rand(1, num_1)-0.5*sigma*ones(1, num_1);
    noise_1_e_x = sigma*rand(1, num_1)-0.5*sigma*ones(1, num_1);
    noise_1_e_y = sigma*rand(1, num_1)-0.5*sigma*ones(1, num_1);
    
    noise_2_s_x = sigma*rand(1, num_2)-0.5*sigma*ones(1, num_2);
    noise_2_s_y = sigma*rand(1, num_2)-0.5*sigma*ones(1, num_2);
    noise_2_e_x = sigma*rand(1, num_2)-0.5*sigma*ones(1, num_2);
    noise_2_e_y = sigma*rand(1, num_2)-0.5*sigma*ones(1, num_2);
    
    noise_3_s_x = sigma*rand(1, num_3)-0.5*sigma*ones(1, num_3);
    noise_3_s_y = sigma*rand(1, num_3)-0.5*sigma*ones(1, num_3);
    noise_3_e_x = sigma*rand(1, num_3)-0.5*sigma*ones(1, num_3);
    noise_3_e_y = sigma*rand(1, num_3)-0.5*sigma*ones(1, num_3);
        
    first_2D_lines = [];
    for j = 1:num_1
        first_2D_lines = [first_2D_lines, [projection3Dpts(K, Rw2c, tw2c, first_3D_lines(1:3, j)); projection3Dpts(K, Rw2c, tw2c, first_3D_lines(4:6, j))]];
    end
    % noise
    if size(first_2D_lines, 2)>0
        first_2D_lines(1, :) = first_2D_lines(1, :) + noise_1_s_x;
        first_2D_lines(2, :) = first_2D_lines(2, :) + noise_1_s_y;
        first_2D_lines(4, :) = first_2D_lines(4, :) + noise_1_e_x;
        first_2D_lines(5, :) = first_2D_lines(5, :) + noise_1_e_y;
    end
    
    
    second_2D_lines = [];
    for j = 1:num_2
        second_2D_lines = [second_2D_lines, [projection3Dpts(K, Rw2c, tw2c, second_3D_lines(1:3, j)); projection3Dpts(K, Rw2c, tw2c, second_3D_lines(4:6, j))]];
    end
    % noise
    if size(second_2D_lines, 2)>0
        second_2D_lines(1, :) = second_2D_lines(1, :) + noise_2_s_x;
        second_2D_lines(2, :) = second_2D_lines(2, :) + noise_2_s_y;
        second_2D_lines(4, :) = second_2D_lines(4, :) + noise_2_e_x;
        second_2D_lines(5, :) = second_2D_lines(5, :) + noise_2_e_y;
    end
    
    third_2D_lines = [];
    for j = 1:num_3
        third_2D_lines = [third_2D_lines, [projection3Dpts(K, Rw2c, tw2c, third_3D_lines(1:3, j)); projection3Dpts(K, Rw2c, tw2c, third_3D_lines(4:6, j))]];
    end
    % noise
    if size(third_2D_lines, 2)>0
        third_2D_lines(1, :) = third_2D_lines(1, :) + noise_3_s_x;
        third_2D_lines(2, :) = third_2D_lines(2, :) + noise_3_s_y;
        third_2D_lines(4, :) = third_2D_lines(4, :) + noise_3_e_x;
        third_2D_lines(5, :) = third_2D_lines(5, :) + noise_3_e_y;
    end
    
    
    %% Step two: outliers
    num_outliers = round(outlierRatio*totalNum);
    line_outliers = [];
    for i = 1:num_outliers
        if ~param.rand_data_gen
            s_x = rand(stream, 1, 1) * 640;
            s_y = rand(stream, 1, 1) * 480;
            e_x = rand(stream, 1, 1) * 640;
            e_y = rand(stream, 1, 1) * 480;
        else
            rng('shuffle');
            s_x = rand(1, 1) * 640;
            s_y = rand(1, 1) * 480;
            e_x = rand(1, 1) * 640;
            e_y = rand(1, 1) * 480;
        end
    
        oneOutlier = [s_x; s_y; 1; e_x; e_y; 1];
    
        line_outliers = [line_outliers, oneOutlier];
    end
        
    disp('Generate number of outliers:');
    disp(num_outliers);
    
    for i = 1:num_outliers
        flag = rem(i,3);
        if flag==1  
            first_2D_lines(:, num_1-floor(i/3)) = line_outliers(:, i);
            gt_correspond_list(1, num_1-floor(i/3)) = 0;
        elseif flag==2
            second_2D_lines(:, num_2-floor(i/3)) = line_outliers(:, i);
            gt_correspond_list(2, num_1+num_2-floor(i/3)) = 0;
        elseif flag==0
            third_2D_lines(:, num_3-floor(i/3)+1) = line_outliers(:, i);
            gt_correspond_list(3, totalNum-floor(i/3)+1) = 0;
        end
    end

    figure(1)
    for j = 1:num_1
        if j>=num_1-num_outliers/3
            plot([first_2D_lines(1, j), first_2D_lines(4, j)], [first_2D_lines(2, j), first_2D_lines(5, j)], 'c');
            hold on
        else
            plot([first_2D_lines(1, j), first_2D_lines(4, j)], [first_2D_lines(2, j), first_2D_lines(5, j)], 'r');
            hold on
        end
    end
    for j = 1:num_2
        if j>=num_2-num_outliers/3
            plot([second_2D_lines(1, j), second_2D_lines(4, j)], [second_2D_lines(2, j), second_2D_lines(5, j)], 'c');
            hold on
        else
            plot([second_2D_lines(1, j), second_2D_lines(4, j)], [second_2D_lines(2, j), second_2D_lines(5, j)], 'g');
            hold on
        end
    end
    for j = 1:num_3
        if j>=num_3-num_outliers/3
            plot([third_2D_lines(1, j), third_2D_lines(4, j)], [third_2D_lines(2, j), third_2D_lines(5, j)], 'c');
            hold on
        else
            plot([third_2D_lines(1, j), third_2D_lines(4, j)], [third_2D_lines(2, j), third_2D_lines(5, j)], 'b');
            hold on
        end
    end
    
    
    %% Step three: normalized lines
    first_2D_lines_norm = [];
    for j = 1:num_1
        first_2D_lines_norm = [first_2D_lines_norm, [normalizePt(K, first_2D_lines(1:3, j)); normalizePt(K, first_2D_lines(4:6, j))]];
    end
    second_2D_lines_norm = [];
    for j = 1:num_2
        second_2D_lines_norm = [second_2D_lines_norm, [normalizePt(K, second_2D_lines(1:3, j)); normalizePt(K, second_2D_lines(4:6, j))]];
    end
    third_2D_lines_norm = [];
    for j = 1:num_3
        third_2D_lines_norm = [third_2D_lines_norm, [normalizePt(K, third_2D_lines(1:3, j)); normalizePt(K, third_2D_lines(4:6, j))]];
    end

    figure(2)
    for j = 1:num_1
        if j>=num_1-num_outliers/3
            plot([first_2D_lines_norm(1, j), first_2D_lines_norm(3, j)], [first_2D_lines_norm(2, j), first_2D_lines_norm(4, j)], 'c');
            hold on
        else
            plot([first_2D_lines_norm(1, j), first_2D_lines_norm(3, j)], [first_2D_lines_norm(2, j), first_2D_lines_norm(4, j)], 'r');
            hold on
        end
    end
    for j = 1:num_2
        if j>=num_2-num_outliers/3
            plot([second_2D_lines_norm(1, j), second_2D_lines_norm(3, j)], [second_2D_lines_norm(2, j), second_2D_lines_norm(4, j)], 'c');
            hold on
        else
            plot([second_2D_lines_norm(1, j), second_2D_lines_norm(3, j)], [second_2D_lines_norm(2, j), second_2D_lines_norm(4, j)], 'g');
            hold on
        end
    end
    for j = 1:num_3
        if j>=num_3-num_outliers/3
            plot([third_2D_lines_norm(1, j), third_2D_lines_norm(3, j)], [third_2D_lines_norm(2, j), third_2D_lines_norm(4, j)], 'c');
            hold on
        else
            plot([third_2D_lines_norm(1, j), third_2D_lines_norm(3, j)], [third_2D_lines_norm(2, j), third_2D_lines_norm(4, j)], 'b');
            hold on
        end
    end
    
    all_2D_lines_norm = [first_2D_lines_norm, second_2D_lines_norm, third_2D_lines_norm];
    
    parallel_line_list = zeros(size(all_2D_lines_norm, 2), 3);
    % backProjPlane's norml
    for j = 1:size(all_2D_lines_norm, 2)
        normal = cross([all_2D_lines_norm(1:2, j); 1], [all_2D_lines_norm(3:4, j); 1]);
        normal = normal / norm(normal);
        parallel_line_list(j, :) = normal';
    end
    
    % uncertainty
    uncertainty = [];
    for j = 1:size(all_2D_lines_norm, 2)
        uncertainty = [uncertainty, line_uncertainty(K, all_2D_lines_norm(1:2, j), all_2D_lines_norm(3:4, j))];
    end
    
end


function [first_3D_lines, second_3D_lines, third_3D_lines, MF_GT] = generate3Dlines(num_1, num_2, num_3, totalNum, param, K)

    %% construct three main direction
    rng('shuffle');
    
    % Step 1: Fixed first direction aligned with gravity (Y axis)
    first_dir = [0; 1.0; 0];  % Y-axis negative direction (aligned with gravity)
    
    % Step 2: Add small noise/perturbation to the first direction
    noise_level = 1e-3;  % Set noise level (e.g., 0.01)
    noise = noise_level * randn(3, 1);  % Generate small random noise
    
    % Add the noise to the first direction
    first_dir_noisy = first_dir + noise;
    
    % Normalize the resulting vector to ensure it's still a unit vector
    first_dir_noisy = first_dir_noisy / norm(first_dir_noisy);
    
    % Step 3: Generate a random vector for the third direction
    r = randn(1, 3);  % Random vector
    third_dir = r(:) / norm(r);  % Normalize to get the third direction
    
    % Step 4: Ensure the third direction is orthogonal to the noisy first direction
    third_dir = third_dir - (first_dir_noisy' * third_dir) * first_dir_noisy;  % Project and subtract
    third_dir = third_dir / norm(third_dir);  % Normalize
    
    % Step 5: Compute the second direction using the cross product of third_dir and first_dir_noisy
    second_dir = cross(third_dir, first_dir_noisy);  % Ensure right-hand system
    
    MF_GT = [first_dir_noisy, second_dir, third_dir];
    
    %% construct 90 randam(repeated) start points in {[-2,2], [-2,2], [4,8]} 
    % random start points
    Lines_1_dir = zeros(6, num_1);
    Lines_2_dir = zeros(6, num_2);
    Lines_3_dir = zeros(6, num_3);
    
    counter = 1;
    while true
        rng('shuffle');
        rand_start_x = -2 + (2+2)*rand;
        rand_start_y = -2 + (2+2)*rand;
        rand_start_z = 4 + (2+2)*rand;
    
        if (rand_start_x/rand_start_z < -K(1,3)/K(1,1)) || (rand_start_x/rand_start_z > K(1,3)/K(1,1)) || ...
           (rand_start_y/rand_start_z < -K(2,3)/K(2,2)) || (rand_start_y/rand_start_z > K(2,3)/K(2,2))
            continue;
        end
    
        rand_start_point = [rand_start_x; rand_start_y; rand_start_z];
    
        stepLength = 4.0;
    
        if counter <= num_1
            end_point = rand_start_point + first_dir_noisy*stepLength;
        elseif counter > num_1 && counter <= num_1+num_2
            end_point = rand_start_point + second_dir*stepLength;
        elseif counter > num_1+num_2 && counter <= totalNum
            end_point = rand_start_point + third_dir*stepLength;
        end
    
        if (end_point(1)/end_point(3) < -K(1,3)/K(1,1)) || (end_point(1)/end_point(3) > K(1,3)/K(1,1)) || ...
           (end_point(2)/end_point(3) < -K(2,3)/K(2,2)) || (end_point(2)/end_point(3) > K(2,3)/K(2,2))
            continue;
        end
    
        i_3D_line_dir = [rand_start_point; end_point];
    
        if counter <= num_1
            Lines_1_dir(:,counter) = i_3D_line_dir;
        elseif counter > num_1 && counter <= num_1+num_2
            Lines_2_dir(:,counter-num_1) = i_3D_line_dir;
        elseif counter > num_1+num_2 && counter <= totalNum
            Lines_3_dir(:,counter-num_1-num_2) = i_3D_line_dir;
        end
    
        if counter == totalNum
            break
        end
        counter = counter + 1;
    end
    
    first_3D_lines = Lines_1_dir;
    second_3D_lines = Lines_2_dir;
    third_3D_lines = Lines_3_dir;

end

function [qs_h] = projection3Dpts(K, Rw2c, tw2c, ps_w)

    col_num = size(ps_w, 2);
    
    % space Pt in Cam coordinate
    tw2c_repmat = repmat(tw2c, 1, col_num);
    
    ps_c = Rw2c*ps_w + tw2c_repmat;
    qs_h = K*ps_c;
    
    for i = 1:col_num
        qs_h(:, i) = [qs_h(1, i)/qs_h(3, i), qs_h(2, i)/qs_h(3, i), 1]';
    end

end 

function normalizedPt = normalizePt(K, ordPt)

    f = (K(1, 1) + K(2, 2)) / 2;
    cu = K(1, 3);
    cv = K(2, 3);
    
    xn = (ordPt(1) - cu) / f;
    yn = (ordPt(2) - cv) / f;
    
    normalizedPt = [xn; yn];

end


function uncertainty = line_uncertainty(K, start_point, end_point)

    start_point_h = [start_point; 1];
    end_point_h = [end_point; 1];
    Sigma_1 = 2 * eye(2,2);
    Sigma_2 = 2 * eye(2,2);
    Sigma_1_h = zeros(3);
    Sigma_1_h(1:2, 1:2) = Sigma_1;
    Sigma_2_h = zeros(3);
    Sigma_2_h(1:2, 1:2) = Sigma_2;
    Sigma_1_h = K \ Sigma_1_h * inv(K)';
    Sigma_2_h = K \ Sigma_2_h * inv(K)';

    l_3d = cross(start_point_h, end_point_h);
    Sigma_3d = skew_symm(end_point_h) * Sigma_1_h * skew_symm(end_point_h)' ...
             + skew_symm(start_point_h) * Sigma_2_h * skew_symm(start_point_h)';

    l_3d_normalized = l_3d / norm(l_3d);
    J = (eye(3) - l_3d_normalized * l_3d_normalized') / norm(l_3d);
    Sigma_3d_normalized = J * Sigma_3d * J';

    Sigma_3d_final = trace(Sigma_3d_normalized);

    uncertainty = 1 / Sigma_3d_final / 10^5;

end

function X_skew = skew_symm(x)

    X_skew = zeros(3, 3);
    X_skew(1, 2) = -x(3);
    X_skew(1, 3) = x(2);
    X_skew(2, 1) = x(3);
    X_skew(2, 3) = -x(1);
    X_skew(3, 1) = -x(2);
    X_skew(3, 2) = x(1);

end
