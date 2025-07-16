function eig_check_pass = Single_Check_Eig(W, param)
    eig_check_pass = [];
    eig_list = [];
    for i = 1:2
        eig_list = [eig_list, eig(value(W(1:3,1:3,i)))];
    end

    for i = 1:2
        if eig_list(3,i) / eig_list(2,i) > param.eigen_thredhold
            eig_check_pass = [eig_check_pass; 1];
        else
            eig_check_pass = [eig_check_pass; 0];
        end
    end
    

end