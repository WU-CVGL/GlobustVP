function sample_pair_ids = SampleLines(image_lines)

sample_pair_ids = generateBin(image_lines);

end


function sample_pair_idxes = generateBin(all_2D_lines)
    
lines = all_2D_lines';
numOfLines = size(lines,1);

histogramLen = 15;

dirHistogram = zeros(histogramLen, 2);
dirHistogram(:,2)=(1:histogramLen)';
dirCell = cell(histogramLen,1);

resolution = pi/histogramLen;
dx = lines(:,3) - lines(:,1);% dx = ex-sx;
dy = lines(:,4) - lines(:,2);% dy = ey-sy;
for lineID = 1: numOfLines
    dir = atan(dy(lineID)/dx(lineID)); %  -pi/2 to pi/2.
    binID = max(ceil((dir+pi/2)/resolution),1);

    dirHistogram(binID, 1) = dirHistogram(binID, 1) + 1; 
    dirCell{binID} = [dirCell{binID}, lineID];
end

dirHistogram = sortrows(dirHistogram, 1);
peakID1 = dirHistogram(end, 2);

for i = 1:histogramLen
    testID = dirHistogram(end-i, 2);
    if abs(testID - peakID1)>=4
        sample_pair_idxes = dirCell{peakID1}; 
        break;
    end
end

end


function s = ComputeIterTimes(conf, N1, N2, K, r, mode)

histogramLen = 180;

num_outlier_aBin = K*r/histogramLen;
if mode == 0
    inlier_ratio1 = 1-num_outlier_aBin/N1;
    inlier_ratio2 = 1-num_outlier_aBin/N2;
    p = inlier_ratio1*inlier_ratio2;
elseif mode == 1
    num_inlier1 = N1-num_outlier_aBin;
    p = nchoosek(num_inlier1, 2)/nchoosek(N1, 2);
elseif mode == 2
    inlier_ratio1 = 1-num_outlier_aBin/N1;
    p = inlier_ratio1;
end

s = log(1-conf)/log(1-p);
s = ceil(s);

end
