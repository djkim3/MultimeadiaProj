% Setup MatConvNet.
run matlab/vl_setupnn ;

% Load a model and upgrade it to MatConvNet current version.
% net = dagnn.DagNN.loadobj(load('Model_mat/imagenet-resnet-50-dag.mat')) ;
% net = dagnn.DagNN.loadobj(load('Model_mat/pascal-fcn8s-dag.mat')) ;
% net = dagnn.DagNN.loadobj(load('Model_mat/fast-rcnn-caffenet-pascal07-dagnn.mat')) ;
net = dagnn.DagNN.loadobj(load('Model_mat/imagenet-googlenet-dag.mat')) ;
% net = dagnn.DagNN.loadobj(load('Model_mat/imagenet-resnet-101-dag.mat')) ;


net.mode = 'test' ;
net.conserveMemory = false;

%INDEX%
% index_name = 'fc1000';% imagenet-resnet-50-dag.mat
% index_name= 'fc7x';%pascal-fcn8s-dag.mat
index_name= 'cls3_fc';%'cls3_pool';%'cls3_fc';%imagenet-googlenet-dag.mat
% index_name= 'fc1000';

net.vars(net.getVarIndex(index_name)).precious = true; % imagenet-resnet-50-dag.mat

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
           % run the CNN
           net.eval({'data', im_}) ;
           featureVector = net.vars(net.getVarIndex(index_name)).value;
%            output = GETOUTPUTS(35)%GETOUTPUTS 
%            featureVector = res(get_num_of_layer).x;
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
           net.eval({'data', im_}) ;
           featureVector = net.vars(net.getVarIndex(index_name)).value;           
           featureVector = featureVector(:).';% transpose
           featureMat2 = [featureMat2; featureVector];
        end
        
        L2Mat=[];
        idxMat=[];% ������ �ε����� ������ mat
        % 2���� featureMat�� ������ L2 norm ��� % D = norm(featurevector1 - featurevector2);
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
        
        
        
        indexPairs = matchFeatures(featureMat1,featureMat2,'maxRatio',0.8);
        fprintf('#. indexPair : %d\n',length(indexPairs(:,1)));
        
        
        
        % L2Mat�� �� �࿡�� ���� ���� ���� ���� ã��
        cnt=0;
        for r=1:length(L2Mat(:,1)) % L2Mat�� ���� ������ŭ ����
            l2_row = L2Mat(r,:);% ���� ����
            [sorted, sort_idx] = sort(l2_row); % ���� ���� ������ ������ ������, 'idx'�� ���� �ε����� ��
            ratio = sorted(1,1)/sorted(1,2);% ���� ���� ���� �� ���� ���� ���� ������ ����.
            
            if ratio < 0.8%1.0 %0.8 % ratio�� 0.8�̸��̸� ��Ī(IFT ����)
%                fprintf('Ratio:%f -> [MATCHING IMG1 patch%d - IMG2 patch%d]\n',ratio,r,sort_idx(1,1));
               cnt=cnt+1;
            end            
        end
        fprintf('# of MACHING : %d\n',cnt);   
        
        % homography �о true matching���� Ȯ�� - �� ������ �Ÿ��� 1 ���ϸ� true��� ��
        H_name = sprintf('./data/Mikolajczyk/%s/H1to%d',img_set,second_img_idx);
        H = dlmread(H_name);
        % kpt ���� txt ����
        kpt1_file_path = sprintf('./patches/%s1_kpt.txt',img_set);
        kpt2_file_path = sprintf('./patches/%s%d_kpt.txt',img_set,second_img_idx);
        kpt1 = dlmread(kpt1_file_path);% img1's kpt info
        kpt2 = dlmread(kpt2_file_path);% img2's kpt info
        
        matchCount=0;
        for r=1:length(L2Mat(:,1)) % L2Mat�� ���� ������ŭ ����, r�� img1�� ��ġ�� �ε���
            l2_row = L2Mat(r,:);% ���� ����
            [sorted, sort_idx] = sort(l2_row); % ���� ���� ������ ������ ������, 'idx'�� ���� �ε����� ��
            img1_patch_idx=r;
            img2_patch_idx=sort_idx(1,1);
%             fprintf('idx1:%d, idx2:%d ',img1_patch_idx-1,img2_patch_idx-1);
            
            %�� ���� ������ ����
            img1_x=kpt1(img1_patch_idx,1);    
            img1_y=kpt1(img1_patch_idx,2);
            img2_x=kpt2(img2_patch_idx,1);
            img2_y=kpt2(img2_patch_idx,2);
            
            %homography ����� ��ǥ ���
            C = H*[img2_x; img2_y; ones(1,1)];
            C(:,1) = C(:,1)/C(3,1);
    
            dist = sqrt(abs(img1_x-C(1,1))*abs(img1_x-C(1,1)) + abs(img1_y-C(2,1))*abs(img1_y-C(2,1))); 
%             fprintf('dist : %d\n',dist);
            if dist < 1.0 % 1.5
                matchCount=matchCount+1;
%                 goodMatch(matchCount,1)= matchIdx;
            end
        end
        fprintf('# of MACHING USING HOMOGRAPHY : %d\n',matchCount);   
        
        
        
        
        
        
        
    end
end