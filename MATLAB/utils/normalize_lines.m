function lines = normalize_lines(lines,fc,cc)

num    = size(lines,1);
inv_fc = [1,1]./fc;
for k = 1:num    
    lines(k, 2:3) = (lines(k, 2:3) - cc).*inv_fc;%start point
    lines(k, 4:5) = (lines(k, 4:5) - cc).*inv_fc;%end point
end
