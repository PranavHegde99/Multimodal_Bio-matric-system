
clear all
clc
close all

% You can customize and fix initial directory paths
TrainDatabasePath = uigetdir('C:\Users\pranav\Desktop\thesis\Desktop\thesisDesktop\thesis folder\train', 'Select train database path');
TestDatabasePath = uigetdir('C:\Users\pranav\Desktop\thesis\Desktop\thesis folder\test', 'Select test database path');
prompt = {'Enter test image name (a number between 1 to 10):'};
dlg_title = 'Input of FLD-Based Face Recognition System';
num_lines= 1;
def = {'1'};

TestImage  = inputdlg(prompt,dlg_title,num_lines,def);
TestImage = strcat(TestDatabasePath,'\',char(TestImage),'.bmp');
im = imread(TestImage);

T = [];
for i = 1 : 50
    
    % I have chosen the name of each image in databases as a corresponding
    % number. However, it is not mandatory!
    str = int2str(i);
    str = strcat('\',str,'.bmp');
    str = strcat(TrainDatabasePath,str);
    
    img = imread(str);
    img = rgb2gray(img);
    
    [irow icol] = size(img);
   
    temp = reshape(img',irow*icol,1);   % Reshaping 2D images into 1D image vectors
    T = [T temp]; % 'T' grows after each turn                    
end

T = double(T);
%T = CreateDatabase(TrainDatabasePath);
Class_number = 10; % Number of classes (or persons)
Class_population = 5; % Number of images in each class
P = Class_population * Class_number; % Total number of training images

%%%%%%%%%%%%%%%%%%%%%%%% calculating the mean image 
m_database = mean(T,2); 

%%%%%%%%%%%%%%%%%%%%%%%% Calculating the deviation of each image from mean image
A = T - repmat(m_database,1,P);

%%%%%%%%%%%%%%%%%%%%%%%% Snapshot method of Eigenface algorithm
L = A'*A; % L is the surrogate of covariance matrix C=A*A'.
[V D] = eig(L); % Diagonal elements of D are the eigenvalues for both L=A'*A and C=A*A'.

%%%%%%%%%%%%%%%%%%%%%%%% Sorting and eliminating small eigenvalues
L_eig_vec = [];
for i = P:-1:Class_number+1
    L_eig_vec = [L_eig_vec V(:,i)];
end

%%%%%%%%%%%%%%%%%%%%%%%% Calculating the eigenvectors of covariance matrix 'C'
V_PCA = A * L_eig_vec; % A: centered image vectors

%%%%%%%%%%%%%%%%%%%%%%%% Projecting centered image vectors onto eigenspace
% Zi = V_PCA' * (Ti-m_database)
ProjectedImages_PCA = [];
for i = 1 : P
    temp = V_PCA'*A(:,i);
    ProjectedImages_PCA = [ProjectedImages_PCA temp]; 
end

%%%%%%%%%%%%%%%%%%%%%%%% Calculating the mean of each class in eigenspace
m_PCA = mean(ProjectedImages_PCA,2); % Total mean in eigenspace
m = zeros(P-Class_number,Class_number); 
Sw = zeros(P-Class_number,P-Class_number); % Initialization os Within Scatter Matrix
Sb = zeros(P-Class_number,P-Class_number); % Initialization of Between Scatter Matrix

for i = 1 : Class_number
    m(:,i) = mean( ( ProjectedImages_PCA(:,((i-1)*Class_population+1):i*Class_population) ), 2 )';    
    
    S  = zeros(P-Class_number,P-Class_number); 
    for j = ( (i-1)*Class_population+1 ) : ( i*Class_population )
        S = S + (ProjectedImages_PCA(:,j)-m(:,i))*(ProjectedImages_PCA(:,j)-m(:,i))';
    end
    
    Sw = Sw + S; % Within Scatter Matrix
    Sb = Sb + Class_number*(m(:,i)-m_PCA) * (m(:,i)-m_PCA)'; % Between Scatter Matrix
end

%%%%%%%%%%%%%%%%%%%%%%%% Calculating Fisher discriminant basis's
% We want to maximise the Between Scatter Matrix, while minimising the
% Within Scatter Matrix. Thus, a cost function J is defined, so that this condition is satisfied.
[J_eig_vec, J_eig_val] = eig(Sb,Sw); % Cost function J = inv(Sw) * Sb
J_eig_vec = fliplr(J_eig_vec);

%%%%%%%%%%%%%%%%%%%%%%%% Eliminating zero eigens and sorting in descend order
for i = 1 : Class_number-1 
    V_Fisher(:,i) = J_eig_vec(:,i); % Largest (C-1) eigen vectors of matrix J
end

%%%%%%%%%%%%%%%%%%%%%%%% Projecting images onto Fisher linear space
% Yi = V_Fisher' * V_PCA' * (Ti - m_database) 
for i = 1 : Class_number*Class_population
    ProjectedImages_Fisher(:,i) = V_Fisher' * ProjectedImages_PCA(:,i);
end
%[m V_PCA V_Fisher ProjectedImages_Fisher] = FisherfaceCore1(T);
%OutputName = Recognition(TestImage, m, V_PCA, V_Fisher, ProjectedImages_Fisher);
Train_Number = size(ProjectedImages_Fisher,2);
%%%%%%%%%%%%%%%%%%%%%%%% Extracting the FLD features from test image
InputImage = imread(TestImage);
temp = InputImage(:,:,1);

[irow icol] = size(temp);
InImage = reshape(temp',irow*icol,1);
Difference = double(InImage)-m_database; % Centered test image
ProjectedTestImage = V_Fisher' * V_PCA' * Difference; % Test image feature vector

%%%%%%%%%%%%%%%%%%%%%%%% Calculating Euclidean distances 
% Euclidean distances between the projected test image and the projection
% of all centered training images are calculated. Test image is
% supposed to have minimum distance with its corresponding image in the
% training database.

Euc_dist = [];
for i = 1 : Train_Number
    q = ProjectedImages_Fisher(:,i);
    temp = ( norm( ProjectedTestImage - q ) )^2;
    Euc_dist = [Euc_dist temp];
end

[Euc_dist_min , Recognized_index] = min(Euc_dist);
OutputName = strcat(int2str(Recognized_index),'.bmp');

SelectedImage = strcat(TrainDatabasePath,'\',OutputName);
SelectedImage = imread(SelectedImage);

imshow(im)
title('Test Image');
figure,imshow(SelectedImage);
title('Equivalent Image');

str = strcat('Matched image is :  ',OutputName);
disp(str)
