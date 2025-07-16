function [vp_solution, correspond_line_id] = Single_Recover_Results(W, param)
    [U,S,V] = svd(W(1:3,1:3,1));
    A = S*V';
    vp_solution = A(1,:);

    correspond_line_id = [];
    for i = 2:size(W,1)/3
        W_block = W(1:3,i*3-2:i*3,1);
        if abs(trace(W_block) - 1) < param.recover_thredhold
            correspond_line_id = [correspond_line_id;i-1];
        end
    end
end