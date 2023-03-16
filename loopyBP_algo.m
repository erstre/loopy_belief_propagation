%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%  Term Project for Probabilistic Graphical Models %%%%%%%%%%%
%%%%%%%                Exercise 6.4 MIT                  %%%%%%%%%%%
%%%%%%%       Streviniotis Errikos AM: 2020039017        %%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc
clear all;
close all;

% Load the images (.bmp files) 
flower = double(imread('source_images/flower.bmp'));
flower_norm = flower / max(flower(:));

background = imread('source_images/background.bmp');
foreground = imread('source_images/foreground.bmp');

number_rows = size(background,1);
number_columns = size(background,2);

% Show the loaded images (.bmp files)
figure(1)
imshow(flower)
figure(2)
imshow(background)
figure(3)
imshow(foreground)

% Find the non-zero values in the .bmp files
[rB, cB] = find(background);
[rF, cF] = find(foreground);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Computation of mean vectors
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Background section
rgb_background  = 0;
for i = 1:1:size(rB,1)
    rgb_background = rgb_background + flower_norm(rB(i), cB(i), 1);
end
rgb1_background_mean = rgb_background / size(rB,1);

rgb_background  = 0;
for i = 1:1:size(rB,1)
    rgb_background = rgb_background + flower_norm(rB(i), cB(i), 2);
end
rgb2_background_mean = rgb_background / size(rB,1);

rgb_background  = 0;
for i = 1:1:size(rB,1)
    rgb_background = rgb_background + flower_norm(rB(i), cB(i), 3);
end
rgb3_background_mean = rgb_background / size(rB,1);

% Computed mean vector for background
rgb_background_mean = [rgb1_background_mean, rgb2_background_mean, rgb3_background_mean]

% Foreground section
rgb_foreground  = 0;
for i = 1:1:size(rF,1)
    rgb_foreground = rgb_foreground + flower_norm(rF(i), cF(i), 1);
end
rgb1_foreground_mean = rgb_foreground / size(rF,1);

rgb_foreground  = 0;
for i = 1:1:size(rF,1)
    rgb_foreground = rgb_foreground + flower_norm(rF(i), cF(i), 2);
end
rgb2_foreground_mean = rgb_foreground / size(rF,1);

rgb_foreground  = 0;
for i = 1:1:size(rF,1)
    rgb_foreground = rgb_foreground + flower_norm(rF(i), cF(i), 3);
end
rgb3_foreground_mean = rgb_foreground / size(rF,1);

% Computed mean vector for foreground
rgb_foreground_mean = [rgb1_foreground_mean, rgb2_foreground_mean, rgb3_foreground_mean]

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Computation of covariance matrix
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Background section
rgb1_background_tmp  = zeros(size(rB,1),1);
rgb2_background_tmp  = zeros(size(rB,1),1);
rgb3_background_tmp  = zeros(size(rB,1),1);
for i = 1:1:size(rB,1)
    rgb1_background_tmp(i) = flower_norm(rB(i), cB(i), 1) - rgb1_background_mean;
    rgb2_background_tmp(i) = flower_norm(rB(i), cB(i), 2) - rgb2_background_mean;
    rgb3_background_tmp(i) = flower_norm(rB(i), cB(i), 3) - rgb3_background_mean;
end
rgb_background_tmp = [rgb1_background_tmp(:), rgb2_background_tmp(:), rgb3_background_tmp(:)].';
rgb_background_tmp_L = rgb_background_tmp * rgb_background_tmp.';

% Computed L matrix for background
rgb_background_L = rgb_background_tmp_L / size(rB,1)

% Foreground section
rgb1_foreground_tmp  = zeros(size(rF,1),1);
rgb2_foreground_tmp  = zeros(size(rF,1),1);
rgb3_foreground_tmp  = zeros(size(rF,1),1);
for i = 1:1:size(rF,1)
    rgb1_foreground_tmp(i) = flower_norm(rF(i), cF(i), 1) - rgb1_foreground_mean;
    rgb2_foreground_tmp(i) = flower_norm(rF(i), cF(i), 2) - rgb2_foreground_mean;
    rgb3_foreground_tmp(i) = flower_norm(rF(i), cF(i), 3) - rgb3_foreground_mean;
end
rgb_foreground_tmp = [rgb1_foreground_tmp(:), rgb2_foreground_tmp(:), rgb3_foreground_tmp(:)].';
rgb_foreground_tmp_L = rgb_foreground_tmp * rgb_foreground_tmp.';

% Computed L matrix for foreground
rgb_foreground_L = rgb_foreground_tmp_L / size(rF,1)


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Sum Product
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

epsilon = 0.01;

% Node potentials 
phi_i = zeros(number_rows,number_columns,2);

for i = 1:1:number_rows
   for j = 1:1:number_columns
       % Part where x = background , phi_i(:,:,1)
       diff = reshape(flower_norm(i,j,:),3,1)-rgb_background_mean';
       phi_i(i,j,1) = 1/((2*pi^(3/2))*(det(rgb_background_L)^0.5)) * exp(-0.5 * diff' * inv(rgb_background_L) * diff) + epsilon;
       
       % Part where x = foreground , phi_i(:,:,2)
       diff = reshape(flower_norm(i,j,:),3,1)-rgb_foreground_mean';
       phi_i(i,j,2) = 1/((2*pi^(3/2))*(det(rgb_foreground_L)^0.5)) * exp(-0.5 * diff' * inv(rgb_foreground_L) * diff) + epsilon;
   end
end

flower_prior = flower_norm;
for i = 1:1:number_rows
   for j = 1:1:number_columns
       if phi_i(i,j,1) >= phi_i(i,j,2)
           pixel = [0 0 0];
       else
           pixel = [255 255 255]; 
       end
       flower_prior(i,j,1) = pixel(1);
       flower_prior(i,j,2) = pixel(2);
       flower_prior(i,j,3) = pixel(3);
   end
end

figure(4)
imshow(flower_prior)   

% We use a clockwise encoding of the messages:
% 1 is for up
% 2 is for right
% 3 is for down
% 4 is for left

msg_t = ones(number_rows, number_columns, 4, 2);
msg_tplus = msg_t;

% 30 iterations procedure
max_itr_num = 30;
x_same = 0.9;
x_diff = 0.1;
psi_ij=[x_same, x_diff;x_diff, x_same];

for itr = 1:1:max_itr_num
   for i =  1:1:number_rows
      for j = 1:1:number_columns
          for msg1 = 1:1:4
              for val = 1:1:2
                 if msg1==1 && i~=1 
                     tmp1 = psi_ij(val,1)*phi_i(i-1,j,1)*msg_t(i-1,j,1,1)*msg_t(i-1,j,4,1)*msg_t(i-1,j,2,1);
                     tmp2 = psi_ij(val,2)*phi_i(i-1,j,2)*msg_t(i-1,j,1,2)*msg_t(i-1,j,4,2)*msg_t(i-1,j,2,2);
                     msg_tplus(i,j,msg1,val) = tmp1 + tmp2;
                 elseif msg1==2 && j~=number_columns
                     tmp1 = psi_ij(val,1)*phi_i(i,j+1,1)*msg_t(i,j+1,1,1)*msg_t(i,j+1,2,1)*msg_t(i,j+1,3,1);
                     tmp2 = psi_ij(val,2)*phi_i(i,j+1,2)*msg_t(i,j+1,1,2)*msg_t(i,j+1,2,2)*msg_t(i,j+1,3,2);
                     msg_tplus(i,j,msg1,val) = tmp1 + tmp2;
                 elseif msg1==3 && i~=number_rows
                     tmp1 = psi_ij(val,1)*phi_i(i+1,j,1)*msg_t(i+1,j,2,1)*msg_t(i+1,j,3,1)*msg_t(i+1,j,4,1);
                     tmp2 = psi_ij(val,2)*phi_i(i+1,j,2)*msg_t(i+1,j,2,2)*msg_t(i+1,j,3,2)*msg_t(i+1,j,4,2);
                     msg_tplus(i,j,msg1,val) = tmp1 + tmp2;
                 elseif msg1==4 && j~=1
                     tmp1 = psi_ij(val,1)*phi_i(i,j-1,1)*msg_t(i,j-1,1,1)*msg_t(i,j-1,3,1)*msg_t(i,j-1,4,1);
                     tmp2 = psi_ij(val,2)*phi_i(i,j-1,2)*msg_t(i,j-1,1,2)*msg_t(i,j-1,3,2)*msg_t(i,j-1,4,2);
                     msg_tplus(i,j,msg1,val) = tmp1 + tmp2;
                 end
              end
          end
       end
   end
   
   %Normalization of the messages
   msg_tplus = msg_tplus./repmat(sum(msg_tplus,4),1,1,1,2); 
   msg_t = msg_tplus;
   
end

% Final Beliefs
belief = ones(number_rows,number_columns, 2);

for i = 1:1:number_rows
   for j = 1:1:number_columns
       for val = 1:1:2
           belief(i,j,val) = phi_i(i,j,val);
           for msg1 = 1:1:4
               belief(i,j,val) = belief(i,j,val)*msg_t(i,j,msg1,val);
           end
       end
   end
end
          
flower_sumproduct = flower_norm;
for i = 1:1:number_rows
   for j = 1:1:number_columns
       if belief(i,j,1) >= belief(i,j,2)
           pixel = [0 0 0];
       else
           pixel = [255 255 255]; 
       end
       flower_sumproduct(i,j,1) = pixel(1);
       flower_sumproduct(i,j,2) = pixel(2);
       flower_sumproduct(i,j,3) = pixel(3);
   end
end

figure(5)
imshow(flower_sumproduct)   
          
