clear all; close all; clc;
%% load data
plane = imread('colorPlane.jpg');
bird = imread('colorBird.jpg');

figure
subplot(1,2,1); imshow(bird)
subplot(1,2,2); imshow(plane)
im_list = {plane,bird};
%% K-MEANS
for i = 1:length(im_list)
    figure
    subplot(2,2,1)
    
    % Horizontal and Vertical Normalized
    [v,h] = meshgrid(1:size(picture,2),1:size(picture,1));
    v_ = h(:);
    h_ = v(:);
    v_norm = ((v_(:) - min(v_(:))) ./ (max(v_(:)) - min(v_(:))));
    h_norm = ((h_(:) - min(h_(:))) ./ (max(h_(:)) - min(h_(:))));
    
    % RGB normalized
    picture = im_list{i};
    red = double(picture(:,:,1)); green = double(picture(:,:,2)); blue = double(picture(:,:,3));
    red_normalized = ((red(:) - min(red(:))) ./ (max(red(:)) - min(red(:))));
    green_normalized = ((green(:) - min(green(:))) ./ (max(green(:)) - min(green(:))));
    blue_normalized = ((blue(:) - min(blue(:))) ./ (max(blue(:)) - min(blue(:))));
    
    % Feature Vector
    feat = [red_normalized green_normalized blue_normalized v_norm h_norm];
    % Kmeans Training
    for k = 2:5
        i = kmeans(feat,k,'replicates',10);
        label = reshape(idx,[size(picture,1),size(picture,2)]);
        grayscale = 255/(k-1);
        for l = 1:k
            label(label == l) = grayscale * (l-1);
        end
        subplot(1,4,k-1)
        imshow(uint8(label))
        title("K = " + num2str(k))
    end
end

%% GMM

for i = 1:length(im_list)
    figure
    subplot(2,2,1)
    
    % Horizontal and Vertical Normalized
    [v,h] = meshgrid(1:size(picture,2),1:size(picture,1));
    v_ = h(:);
    h_ = v(:);
    v_norm = ((v_(:) - min(v_(:))) ./ (max(v_(:)) - min(v_(:))));
    h_norm = ((h_(:) - min(h_(:))) ./ (max(h_(:)) - min(h_(:))));
    
    % RGB Normalized
    picture = im_list{i};
    red = double(picture(:,:,1));
    green = double(picture(:,:,2));
    blue = double(picture(:,:,3));
    red_normalized = ((red(:) - min(red(:))) ./ (max(red(:)) - min(red(:))));
    green_normalized = ((green(:) - min(green(:))) ./ (max(green(:)) - min(green(:))));
    blue_normalized = ((blue(:) - min(blue(:))) ./ (max(blue(:)) - min(blue(:))));
    

    % Feature Vector
    feat = [red_normalized green_normalized blue_normalized v_norm h_norm];
        for k = 2:5
            GMM = fitgmdist(feat,k,'Options',statset('MaxIter',500),'Replicates',10);
            idx = cluster(GMM,feat);
            label = reshape(idx,[size(picture,1),size(picture,2)]);
            grayscale = 255/(k-1);
            for l = 1:k
                label(label == l) = grayscale * (l-1);
            end
            subplot(1,4,k-1)
            imshow(uint8(label))
            title("K = " + num2str(k))
        end
end