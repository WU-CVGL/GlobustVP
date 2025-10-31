function SDPSolver = GetSDPSolver(size_x, line_current_size, param)
    X = sdpvar(size_x, size_x, 2);
    Constraints = [];

    % ---- theta = theta^2 ----
    for point_i = 1:2
        for line_i = 1:line_current_size
            id = line_i * 3;
            Constraints = [Constraints, X(1:3, id+1:id+3, point_i) == X(id+1:id+3, id+1:id+3, point_i)];
        end
    end

    % ---- trace == 1, w >= 0 ----
    for point_i = 1:2
        Constraints = [Constraints, X(:,:,point_i) >= 0];
        Constraints = [Constraints, trace(X(1:3,1:3,point_i)) == 1];
    end
    
    % ---- each block is symmetric ----
    for point_i = 1:2
        for line_i = 2:line_current_size
            for line_j = line_i + 1:line_current_size+1
                sub_i_dense = line_i * 3 - 3;
                sub_j_dense = line_j * 3 - 3;
                Constraints = [Constraints, X(sub_i_dense + 1:sub_i_dense + 3, sub_j_dense + 1:sub_j_dense + 3, point_i) == X(sub_i_dense + 1:sub_i_dense + 3, sub_j_dense + 1:sub_j_dense + 3, point_i)'];
            end
        end
    end

    % ---- sum theta == 1 ----
    for line_i = 1:line_current_size
        id = line_i * 3;
        Constraints_single = 0;
        for point_i = 1:2
            Constraints_single = Constraints_single + X(1:3, id+1:id+3, point_i);
        end
        Constraints = [Constraints, Constraints_single == X(1:3,1:3,1)];
    end

    Constraints = [Constraints, X(1:3,1:3,1) == X(1:3,1:3,2)];

    C_sdp = sdpvar(size_x, size_x, 2);
    Objective = trace(C_sdp(:,:,1) * X(:,:,1)) + trace(C_sdp(:,:,2) * X(:,:,2));
    ops = sdpsettings('solver','mosek','verbose',1,'debug',1);
    SDPSolver = optimizer(Constraints,Objective,ops,C_sdp,X);
end