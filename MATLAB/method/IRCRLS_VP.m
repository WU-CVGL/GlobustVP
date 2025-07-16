function [optimal, solution_list, correspond_list, total_iter] = IRCRLS_VP(parallel_line_list, all_2D_lines_norm, param, uncertainty)
    
    if param.fast_solver
        largest_bin_idxes = SampleLines(all_2D_lines_norm);
        [optimal, solution_list, correspond_list, total_iter] = fast_sdp_solver(parallel_line_list, uncertainty, ...
            largest_bin_idxes, param);
    else
        [optimal, solution_list, correspond_list, total_iter] = iterative_sdp_solver(parallel_line_list, uncertainty, param);
    end
end


function [optimal, solution_list, correspond_list, optimal_counter] = fast_sdp_solver(parallel_line_list, uncertainty, largest_bin_idxes, param)
    yalmip('clear');

    solution_list = [];
    correspond_list = [];
    line_id_pool = true(param.line_num, 1);
    line_reverse_id_pool = linspace(1, param.line_num, param.line_num);
    stop_opt = true;
    optimal_counter = 0;
    while stop_opt
        optimal_counter = optimal_counter + 1;
        line_current = parallel_line_list(line_id_pool, :);
        uncertainty_current = uncertainty(line_id_pool);
        line_current_size = sum(line_id_pool);

        if line_current_size < 3 && size(solution_list, 1) < param.vanishing_point_num
            solution_list = [];
            correspond_list = [];
            line_id_pool = true(param.line_num, 1);
            line_reverse_id_pool = linspace(1, param.line_num, param.line_num);
            continue;
        elseif line_current_size < 3 && size(solution_list, 1) == param.vanishing_point_num
            optimal = true;
            return;
        end

        if isempty(solution_list)
            sample_num = min(length(largest_bin_idxes), param.max_line_processing);
            rand_id = largest_bin_idxes(randperm(length(largest_bin_idxes), sample_num));
            line_current = line_current(rand_id, :);
            line_current_size = sample_num;
            uncertainty_current = uncertainty_current(rand_id);
        else
            sample_num = min(param.max_line_processing, line_current_size); %min(param.max_line_processing, sample_num);
            rand_id = randperm(line_current_size, sample_num);
            line_current = line_current(rand_id, :);
            line_current_size = sample_num;
            uncertainty_current = uncertainty_current(rand_id);
        end

        SDPSolver = GetSDPSolver(3 * line_current_size + 3, line_current_size, param);

        C = zeros(3 * line_current_size + 3, 3 * line_current_size + 3, 2);
        for line_i = 1:line_current_size
            line_single = line_current(line_i, :);
            uncertainty_single = uncertainty_current(line_i);
            id = line_i * 3;
            C(1:3, id+1:id+3, 1) = 0.5 * (line_single' * line_single) * uncertainty_single;
            C(id+1:id+3, 1:3, 1) = 0.5 * (line_single' * line_single) * uncertainty_single;
            C(1:3, id+1:id+3, 2) = 0.5 * param.c.^2 * eye(3) * uncertainty_single;
            C(id+1:id+3, 1:3, 2) = 0.5 * param.c.^2 * eye(3) * uncertainty_single;
        end
        W = SDPSolver(C);

        % ------ save results -----
        eig_check = Single_Check_Eig(value(W), param);
        if min(eig_check) == 0
            optimal = false;
            continue;
        end

        [vp_solution, correspond_line_id] = Single_Recover_Results(value(W), param);

        if size(solution_list, 1) == 1 && (abs(acosd(dot(solution_list,vp_solution')) - 90) > 90-acosd(param.c))
            solution_list = [];
            correspond_list = [];
            line_id_pool = true(param.line_num, 1);
            line_reverse_id_pool = linspace(1, param.line_num, param.line_num);
            continue;
        elseif size(solution_list, 1) == 2 && (abs(dot(solution_list(1,:),vp_solution')) > 1e-3 || abs(dot(solution_list(2,:),vp_solution')) > 1e-3)
            vp_solution = cross(solution_list(1,:), solution_list(2,:));
        end

        solution_list = [solution_list;vp_solution];

        % ------ update iteration -----
        if sample_num == line_current_size
            line_all = parallel_line_list(line_id_pool,:);
            correspond_line_id = find(abs(line_all * vp_solution') < param.c);
            original_id = line_reverse_id_pool(1, correspond_line_id);

            if size(solution_list, 1) == 1
                [left_lines, peak_line_idx, peak_n1, peak_n2] = find_peak_intervals(vp_solution, parallel_line_list);
                line_id_pool = false(param.line_num, 1);
                line_id_pool(peak_line_idx',:) = 1;
                line_id_pool(original_id',:) = 0;
                line_reverse_id_pool = find(line_id_pool==1)';
            else
                line_id_pool(original_id',:) = 0;
                line_reverse_id_pool = find(line_id_pool==1)';
            end

            corr = zeros(1, param.line_num);
            corr(original_id) = 1;
            correspond_list = [correspond_list; corr];

        else
            original_id = line_reverse_id_pool(1,correspond_line_id);
            line_id_pool(original_id',:) = 0;
            line_reverse_id_pool = find(line_id_pool==1)';

            corr = zeros(1, param.line_num);
            corr(original_id) = 1;
            correspond_list = [correspond_list; corr];
        end

        if param.vanishing_point_num == 3
            if size(solution_list, 1) == param.vanishing_point_num
                if (det(solution_list) < 0)
                    solution_list(1,:) = -solution_list(1,:);
                end
                [U,~,V] = svd(solution_list);
                solution_list = U*V';
                if norm(round(solution_list * solution_list') - eye(3)) == 0
                    optimal = true;
                    return;
                else
                    optimal_counter = optimal_counter + 1;
                    solution_list = [];
                    correspond_list = [];
                    line_id_pool = true(param.line_num,1);
                    line_reverse_id_pool = linspace(1,param.line_num,param.line_num);
                    continue;
                end
            end
        else
            if size(solution_list, 1) == 2
                optimal = true;
                return;
            end
        end
    end
end


function [optimal, solution_list, correspond_list, optimal_counter] = iterative_sdp_solver(parallel_line_list, uncertainty, param)
    yalmip('clear');

    solution_list = [];
    correspond_list = [];
    line_id_pool = true(param.line_num, 1);
    line_reverse_id_pool = linspace(1, param.line_num, param.line_num);
    stop_opt = true;
    optimal_counter = 0;
    while stop_opt
        optimal_counter = optimal_counter + 1;
        line_current = parallel_line_list(line_id_pool, :);
        uncertainty_current = uncertainty(line_id_pool);
        line_current_size = sum(line_id_pool);

        if line_current_size < 3 && size(solution_list, 1) < param.vanishing_point_num
            solution_list = [];
            correspond_list = [];
            line_id_pool = true(param.line_num, 1);
            line_reverse_id_pool = linspace(1, param.line_num, param.line_num);
            continue;
        elseif line_current_size < 3 && size(solution_list, 1) == param.vanishing_point_num
            optimal = true;
            return;
        end

        if isempty(solution_list)
            sample_num = param.max_line_processing;
            rand_id = randperm(line_current_size, param.max_line_processing);
            line_current = line_current(rand_id, :);
            line_current_size = param.max_line_processing;
            uncertainty_current = uncertainty_current(rand_id);
        else
            sample_num = min(param.max_line_processing, line_current_size);
            rand_id = randperm(line_current_size, sample_num);
            line_current = line_current(rand_id, :);
            line_current_size = sample_num;
            uncertainty_current = uncertainty_current(rand_id);
        end

        [SDPSolver] = GetSDPSolver(3 * line_current_size + 3, line_current_size, param);

        C = zeros(3 * line_current_size + 3, 3 * line_current_size + 3, 2);
        for line_i = 1:line_current_size
            line_single = line_current(line_i, :);
            uncertainty_single = uncertainty_current(line_i);
            id = line_i * 3;
            C(1:3, id+1:id+3, 1) = 0.5 * (line_single' * line_single) * uncertainty_single;
            C(id+1:id+3, 1:3, 1) = 0.5 * (line_single' * line_single) * uncertainty_single;
            C(1:3, id+1:id+3, 2) = 0.5 * param.c.^2 * eye(3) * uncertainty_single;
            C(id+1:id+3, 1:3, 2) = 0.5 * param.c.^2 * eye(3) * uncertainty_single;
        end
        W = SDPSolver(C);

        % ------ save results -----
        eig_check = Single_Check_Eig(value(W), param);
        if min(eig_check) == 0
            optimal = false;
            continue;
        end

        [vp_solution, correspond_line_id] = Single_Recover_Results(value(W), param);

        if size(solution_list, 1) == 1 && (abs(acosd(dot(solution_list,vp_solution')) - 90) > 90-acosd(param.c))
            solution_list = [];
            correspond_list = [];
            line_id_pool = true(param.line_num, 1);
            line_reverse_id_pool = linspace(1, param.line_num, param.line_num);
            continue;
        elseif size(solution_list, 1) == 2 && (abs(dot(solution_list(1,:),vp_solution')) > 1e-3 || abs(dot(solution_list(2,:),vp_solution')) > 1e-3)
            vp_solution = cross(solution_list(1,:), solution_list(2,:));
        end

        solution_list = [solution_list;vp_solution];

        % ------ update iteration -----
        if sample_num == line_current_size
            line_all = parallel_line_list(line_id_pool,:);
            correspond_line_id = find(abs(line_all * vp_solution') < param.c);
            original_id = line_reverse_id_pool(1, correspond_line_id);
            line_id_pool(original_id',:) = 0;
            line_reverse_id_pool = find(line_id_pool==1)';

            corr = zeros(1, param.line_num);
            corr(original_id) = 1;
            correspond_list = [correspond_list; corr];

        else
            original_id = line_reverse_id_pool(1,correspond_line_id);
            line_id_pool(original_id',:) = 0;
            line_reverse_id_pool = find(line_id_pool==1)';

            corr = zeros(1, param.line_num);
            corr(original_id) = 1;
            correspond_list = [correspond_list; corr];
        end

        if param.vanishing_point_num == 3
            if size(solution_list, 1) == param.vanishing_point_num
                if (det(solution_list) < 0)
                    solution_list(1,:) = -solution_list(1,:);
                end
                [U,~,V] = svd(solution_list);
                solution_list = U*V';
                if norm(round(solution_list * solution_list') - eye(3)) == 0
                    optimal = true;
                    return;
                else
                    optimal_counter = optimal_counter + 1;
                    solution_list = [];
                    correspond_list = [];
                    line_id_pool = true(param.line_num,1);
                    line_reverse_id_pool = linspace(1,param.line_num,param.line_num);
                    continue;
                end
            end
        else
            if size(solution_list, 1) == 2
                optimal = true;
                return;
            end
        end
    end
end


function [left_lines, peak_line_idx, peak_n1, peak_n2] = find_peak_intervals(d, normals)
    d = d / norm(d); 
    
    null_vectors = null(d);
    n1 = null_vectors(:,1);
    n2 = cross(d, n1);
    
    num_samples = 90;
    bin_counts = zeros(1, num_samples);
    bin_ids = cell(1, num_samples);
    bin_n1 = zeros(3, num_samples);
    bin_n2 = zeros(3, num_samples);
    
    for angle = 0:num_samples-1
        theta = deg2rad(angle);
        
        R = axang2rotm([d, theta]);
        
        n1_rotated = R * n1;
        
        n2_rotated = cross(d, n1_rotated);
        
        for i = 1:size(normals, 1)
            cos_theta_n1 = dot(normals(i,:), n1_rotated) / (norm(normals(i,:)) * norm(n1_rotated));
            angle_n1 = acosd(cos_theta_n1);
            
            cos_theta_n2 = dot(normals(i,:), n2_rotated) / (norm(normals(i,:)) * norm(n2_rotated));
            angle_n2 = acosd(cos_theta_n2);
            
            if abs(angle_n1 - 90) <= 0.5 || abs(angle_n2 - 90) <= 0.5
                bin_counts(angle + 1) = bin_counts(angle + 1) + 1;
                bin_ids{angle + 1} = [bin_ids{angle + 1}, i];
            end
        end
        bin_n1(:, angle + 1) = n1_rotated;
        bin_n2(:, angle + 1) = n2_rotated;
    end
    
    [peak_val, peak_idx] = max(bin_counts);
    peak_line_idx = bin_ids{peak_idx};
    peak_n1 = bin_n1(:, peak_idx);
    peak_n2 = bin_n2(:, peak_idx);
    left_lines = normals(peak_line_idx, :);
end