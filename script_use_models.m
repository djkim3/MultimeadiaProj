% Setup MatConvNet.
run matlab/vl_setupnn ;

% Load a model and upgrade it to MatConvNet current version.
% net = load('Model_mat/imagenet-vgg-m.mat') ; % ��Ʈ��ũ �� ����
% net = load('Model_mat/imagenet-vgg-m-128.mat') ;
% net = load('Model_mat/imagenet-vgg-m-1024.mat') ;
net = load('Model_mat/imagenet-vgg-m-2048.mat') ;
 
% net = load('Model_mat/imagenet-caffe-alex.mat') ;
net = vl_simplenn_tidy(net) ;
get_num_of_layer = 20;%42;%36;%20; % VGG : (fc7), (fc8) %��Ʈ��Ʈ���� ���� ���̾��� �ε���

imgSetName = {'bark','bikes','boat','bricks','cars','graffiti','trees','ubc'}; % Mikolajczyk Data Set

path = './patches/'; % ��ġ���� �ִ� ���̽� ���� �̸�

for img_set_idx=1:length(imgSetName)
    img_set = char(imgSetName(img_set_idx));% �̹��� ���� �̸��� ����
    %disp(img_set);% �̹��� �� �̸� �׽�Ʈ ���
    patch_folder_path = sprintf('%s%s',path,img_set);%strcat(path,img_set);% �̹��� ��ġ�� �ִ� ������ ��� ����
    
    for second_img_idx =2:6 % ���� ����� �Ǵ� �̹����� �ε���
        fisrt_patch_folder_path = sprintf('%s1/', patch_folder_path);% ��ġ ������ ��ο� �̹��� �ε����� �߰�
        second_patch_folder_path = sprintf('%s%d/', patch_folder_path,second_img_idx); % ��ġ ������ ��ο� �̹��� �ε����� �߰�
        disp(fisrt_patch_folder_path);
        disp(second_patch_folder_path);
        
        fileList1 = dir(fisrt_patch_folder_path);
        numFiles1 = length(fileList1)-2; % fileList.name�� �ε��� 3������ ���� �̸� ����. �׷��� 2�� ����
        fileList2 = dir(second_patch_folder_path);
        numFiles2 = length(fileList2)-2; % fileList.name�� �ε��� 3������ ���� �̸� ����. �׷��� 2�� ����
        fprintf('# of IMG in 1st Folder : %d\n# of IMG in 2nd Folder : %d\n', numFiles1,numFiles2);
                
        % �� �̹����� ��ġ�� feature vector�� ������ ū Mat 2�� �ʿ� - featureMat1,featureMat2
        featureMat1=[];
        featureMat2=[];
        
        for idx=1:numFiles1
           fileName = strcat(fisrt_patch_folder_path,fileList1(idx+2).name);
           img = imread(fileName);
%            imshow(img);
           im_ = single(img) ; % note: 255 range
           im_ = imresize(im_, net.meta.normalization.imageSize(1:2)) ;% �̹����� 224x224�� ��������
           im_ = bsxfun(@minus,im_,net.meta.normalization.averageImage) ;%(A - mean(A))
           
           % Run the CNN.
           res = vl_simplenn(net, im_) ;
           featureVector = res(get_num_of_layer).x;
           featureVector = featureVector (:).';% transpose
           featureMat1 = [featureMat1; featureVector];
        end
        
        
        for idx=1:numFiles2
           fileName = strcat(second_patch_folder_path,fileList2(idx+2).name);
           img = imread(fileName);
%            imshow(img);
           im_ = single(img) ; % note: 255 range
           im_ = imresize(im_, net.meta.normalization.imageSize(1:2)) ;% �̹����� 224x224�� ��������
           im_ = bsxfun(@minus,im_,net.meta.normalization.averageImage) ;%(A - mean(A))
           
           % Run the CNN.
           res = vl_simplenn(net, im_) ;
           featureVector = res(get_num_of_layer).x;
           featureVector = featureVector(:).';% transpose
           featureMat2 = [featureMat2; featureVector];
        end
        
        L2Mat=[]; % 2���� featureMat�� ������ L2 norm ��� % D = norm(featurevector1 - featurevector2);
        for idx1=1:length(featureMat1(:,1)) % IMG1�� patch ������ŭ ����
            L2=[];
            f1 = featureMat1(idx1,:); % idx1��° ��
%             f1 = f1'; % transpose
            for idx2=1:length(featureMat2(:,1)) % IMG2�� patch ������ŭ ����
                % compute L2 norm        
                f2 = featureMat2(idx2,:); % idx2��° �� vector      
%                 f2 = f2';
                    
                V = f1 - f2;
                D = sqrt(V * V');

%                 D = norm(f1 - f2); % �� feature ������ L2 distance�� ���
                L2 = [L2,D]; 
            end            
            L2Mat = [L2Mat; L2];
        end
        
        % L2Mat�� �� �࿡�� ���� ���� ���� ���� ã��
        cnt=0;
        for r=1:length(L2Mat(:,1)) % L2Mat�� ���� ������ŭ ����
            l2_row = L2Mat(r,:);% ���� ����
            [sorted, sort_idx] = sort(l2_row); % ���� ���� ������ ������ ������, 'idx'�� ���� �ε����� ��
            ratio = sorted(1,1)/sorted(1,2);% ���� ���� ���� �� ���� ���� ���� ������ ����.
            
            if ratio < 0.8%1.0 %0.8 % ratio�� 0.8�̸��̸� ��Ī(SIFT ����)
%                fprintf('Ratio:%f -> [MATCHING IMG1 patch%d - IMG2 patch%d]\n',ratio,r,sort_idx(1,1));
               cnt=cnt+1;
            end            
        end
        fprintf('# of MACHING : %d\n',cnt);  
    end
end