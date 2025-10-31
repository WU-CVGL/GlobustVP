function [norm_lines, line_segments] = Real_GenerateData(lines_Old, fc, cc)

norm_lines = normalize_lines(lines_Old,fc,cc);

num_lines = size(lines_Old, 1);

line_segments = zeros(num_lines, 19);
for li = 1:size(line_segments, 1)
    p1 = [lines_Old(li, 2), lines_Old(li, 3), 1];
    p2 = [lines_Old(li, 4), lines_Old(li, 5), 1];
    centroid = 0.5 * (p1 + p2);
    normal = cross(p1, p2);
    normal = normal / norm(normal(1:2));
    line_segments(li, 1:3) = p1;
    line_segments(li, 4:6) = p2;
    line_segments(li, 7:9) = normal;
    line_segments(li, 10:12) = centroid;
    backProjPlane_normals = cross([norm_lines(li, 2:3), 1], [norm_lines(li, 4:5), 1]);
    backProjPlane_normals = backProjPlane_normals / norm(backProjPlane_normals);
    line_segments(li, 13:15) = backProjPlane_normals;
end
end

