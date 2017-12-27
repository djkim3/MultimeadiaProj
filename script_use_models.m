% Setup MatConvNet.
run matlab/vl_setupnn ;

% Load a model and upgrade it to MatConvNet current version.
% net = load('Model_mat/imagenet-vgg-m.mat') ; % 네트워크 모델 설정
% net = load('Model_mat/imagenet-vgg-m-128.mat') ;
% net = load('Model_mat/imagenet-vgg-m-1024.mat') ;
net = load('Model_mat/imagenet-vgg-m-2048.mat') ;
 
% net = load('Model_mat/imagenet-caffe-alex.mat') ;
net = vl_simplenn_tidy(net) ;
get_num_of_layer = 20;%42;%36;%20; % VGG : (fc7), (fc8) %네트워트에서 받을 레이어의 인덱스

imgSetName = {'bark','bikes','boat','bricks','cars','graffiti','trees','ubc'}; % Mikolajczyk Data Set

path = './patches/'; % 패치들이 있는 베이스 폴더 이름

for img_set_idx=1:length(imgSetName)
    img_set = char(imgSetName(img_set_idx));% 이미지 셋의 이름을 받음
    %disp(img_set);% 이미지 셋 이름 테스트 출력
    patch_folder_path = sprintf('%s%s',path,img_set);%strcat(path,img_set);% 이미지 패치가 있는 폴더의 경로 지정
    
    for second_img_idx =2:6 % 비교할 대상이 되는 이미지의 인덱스
        fisrt_patch_folder_path = sprintf('%s1/', patch_folder_path);% 패치 폴더의 경로에 이미지 인덱스를 추가
        second_patch_folder_path = sprintf('%s%d/', patch_folder_path,second_img_idx); % 패치 폴더의 경로에 이미지 인덱스를 추가
        disp(fisrt_patch_folder_path);
        disp(second_patch_folder_path);
        
        fileList1 = dir(fisrt_patch_folder_path);
        numFiles1 = length(fileList1)-2; % fileList.name의 인덱스 3번부터 파일 이름 나옴. 그래서 2개 줄임
        fileList2 = dir(second_patch_folder_path);
        numFiles2 = length(fileList2)-2; % fileList.name의 인덱스 3번부터 파일 이름 나옴. 그래서 2개 줄임
        fprintf('# of IMG in 1st Folder : %d\n# of IMG in 2nd Folder : %d\n', numFiles1,numFiles2);
                
        % 두 이미지의 패치의 feature vector를 저장할 큰 Mat 2개 필요 - featureMat1,featureMat2
        featureMat1=[];
        featureMat2=[];
        
        for idx=1:numFiles1
           fileName = strcat(fisrt_patch_folder_path,fileList1(idx+2).name);
           img = imread(fileName);
%            imshow(img);
           im_ = single(img) ; % note: 255 range
           im_ = imresize(im_, net.meta.normalization.imageSize(1:2)) ;% 이미지를 224x224로 리사이즈
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
           im_ = imresize(im_, net.meta.normalization.imageSize(1:2)) ;% 이미지를 224x224로 리사이즈
           im_ = bsxfun(@minus,im_,net.meta.normalization.averageImage) ;%(A - mean(A))
           
           % Run the CNN.
           res = vl_simplenn(net, im_) ;
           featureVector = res(get_num_of_layer).x;
           featureVector = featureVector(:).';% transpose
           featureMat2 = [featureMat2; featureVector];
        end
        
        L2Mat=[]; % 2개의 featureMat를 가지고 L2 norm 계산 % D = norm(featurevector1 - featurevector2);
        for idx1=1:length(featureMat1(:,1)) % IMG1의 patch 개수만큼 루프
            L2=[];
            f1 = featureMat1(idx1,:); % idx1번째 행
%             f1 = f1'; % transpose
            for idx2=1:length(featureMat2(:,1)) % IMG2의 patch 개수만큼 루프
                % compute L2 norm        
                f2 = featureMat2(idx2,:); % idx2번째 행 vector      
%                 f2 = f2';
                    
                V = f1 - f2;
                D = sqrt(V * V');

%                 D = norm(f1 - f2); % 두 feature 사이의 L2 distance를 계산
                L2 = [L2,D]; 
            end            
            L2Mat = [L2Mat; L2];
        end
        
        % L2Mat의 각 행에서 가장 값이 작은 것을 찾음
        cnt=0;
        for r=1:length(L2Mat(:,1)) % L2Mat의 행의 개수만큼 루프
            l2_row = L2Mat(r,:);% 행을 받음
            [sorted, sort_idx] = sort(l2_row); % 받은 행을 오름차 순으로 정렬함, 'idx'엔 원래 인덱스가 들어감
            ratio = sorted(1,1)/sorted(1,2);% 가장 작은 값과 그 다음 작은 값의 비율을 구함.
            
            if ratio < 0.8%1.0 %0.8 % ratio가 0.8미만이면 매칭(SIFT 기준)
%                fprintf('Ratio:%f -> [MATCHING IMG1 patch%d - IMG2 patch%d]\n',ratio,r,sort_idx(1,1));
               cnt=cnt+1;
            end            
        end
        fprintf('# of MACHING : %d\n',cnt);  
    end
end