function maxima = strict_local_maxima(probmap, threshold, gaussian_std)

gaussianpm = gaussian_filter(probmap,gaussian_std);

image2 = ordfilt2(gaussianpm,9,true(3),'symmetric');

gaussianpmth = gaussianpm > threshold;%sets all values grater then 
%gaussianpm to 1 and all other values 0

equalval = (image2 == gaussianpm);%sets all values that are equal to 1 
%after filtering and before filtering equal
indicator = ( equalval & gaussianpmth);%sets all values that are equal to 

[row_coords, col_coords] = find(indicator == 1);

maxima = [col_coords'; row_coords'];

end