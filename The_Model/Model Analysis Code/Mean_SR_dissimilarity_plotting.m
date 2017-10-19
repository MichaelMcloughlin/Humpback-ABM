% Script Author and Affiliation: Luca Lamoni, University of St Andrews
% Paper title: “Using agent-based models to understand the role of individuals in the song evolution of humpback whales ( _Megaptera novaeangliae_ )”
% Please provide comments and questions at ll42@st-andrews.ac.uk

% We use this script to plot the mean SR dissimilarity

%% We lead the first file of the folder just to get the Index
load('Data_0_Mig0.mat') 

%% Create sparse matrices to filter selectively the SR dissimilarity values from the 2 populations

DIST_BREED1t = zeros(30,30); % in this example the total population of agents is 30
DIST_BREED1t(1:15,1:15) = 1; % population 1, in this example is composed by 15 agents 
DIST_BREED2t = zeros(30,30);
DIST_BREED2t(16:30,16:30) = 1; % population 2, in this example is composed by 15 agents
DIST_BTWt = zeros(30,30);
DIST_BTWt(1:15,16:30 )=1; % We use this matrix to compare the SR dissimilarity values between the 2 sub-populations

%% Allocate space for the arrays that will be created in the loop
SUB_POP_0 = zeros(length(Index),5); % 5 in this case is representing the number of files Data_x_Mig0.mat present in the folder, x being the number of files
SUB_POP_1 = SUB_POP_0;
Sub_pop_BTW_0 =SUB_POP_0;
Sub_pop_BTW_1 = SUB_POP_0;
SUB_POP1_0 = SUB_POP_0;
SUB_POP1_1 = SUB_POP_0;
SUB_POP2_0 = SUB_POP_0;
SUB_POP2_1 = SUB_POP_0;

%% With this loop we calculate the mean SR dissimilarity at each iteration both between and within sub-population 1 and 2
for i = 1:5 %1 to total number of files present in the folder relative to one migration
    for ii = Index % number of the iterations analysed
    
    %% FIRST MIGRATION
    
    
    %% Load the Data_i_Mig0.mat
    load(['Data_' num2str(i-1) '_Mig0.mat']);  
    
    %% mean SR dissimilarity within sub-population 1 and sub-population 2
    SUB_POP_0_temp = squeeze(SUB_d(ii,:,:)).*(DIST_BREED1t+DIST_BREED2t);
    SUB_POP_0(ii,i) = sum(sum(triu(SUB_POP_0_temp)))/(sum(sum(triu(DIST_BREED1t+DIST_BREED2t)-eye(size(DIST_BREED1t)))));
    
    %% mean SR dissimilarity between sub-population 1 and sub-population 2
    Sub_pop_BTWt = squeeze(SUB_d(ii,:,:)).*DIST_BTWt;
    Sub_pop_BTW_0(ii,i) = sum(sum(triu(Sub_pop_BTWt)))/sum(sum(DIST_BTWt));
    
    %% mean SR dissimilarity within population 1
    SUB_POP1_0_temp = squeeze(SUB_d(ii,:,:)).*(DIST_BREED1t);
    SUB_POP1_0(ii,i) = sum(sum(triu(SUB_POP1_0_temp)))/(sum(sum(triu(DIST_BREED1t))));
    
    %% mean SR dissimilarity within population 2
    SUB_POP2_0_temp = squeeze(SUB_d(ii,:,:)).*(DIST_BREED2t);
    SUB_POP2_0(ii,i) = sum(sum(triu(SUB_POP2_0_temp)))/(sum(sum(triu(DIST_BREED2t))));
    
    %% SECOND MIGRATION
    
    %% Load the Data_i_Mig1.mat
    load(['Data_' num2str(i-1) '_Mig1.mat'])
    
    %% mean SR dissimilarity within sub-population 1 and sub-population 2
    SUB_POP_1_temp = squeeze(SUB_d(ii,:,:)).*(DIST_BREED1t+DIST_BREED2t);
    SUB_POP_1(ii,i) = sum(sum(triu(SUB_POP_1_temp)))/(sum(sum(triu(DIST_BREED1t+DIST_BREED2t)-eye(size(DIST_BREED1t)))));
    
    %% mean SR dissimilarity between sub-population 1 and sub-population 2
    Sub_pop_BTWt = squeeze(SUB_d(ii,:,:)).*DIST_BTWt;
    Sub_pop_BTW_1(ii,i) = sum(sum(triu(Sub_pop_BTWt)))/sum(sum(DIST_BTWt));
    
    %% mean SR dissimilarity within population 1
    SUB_POP1_1_temp = squeeze(SUB_d(ii,:,:)).*(DIST_BREED1t);
    SUB_POP1_1(ii,i) = sum(sum(triu(SUB_POP1_1_temp)))/(sum(sum(triu(DIST_BREED1t))));
    
    %% mean SR dissimilarity within population 2
    SUB_POP2_1_temp = squeeze(SUB_d(ii,:,:)).*(DIST_BREED2t);
    SUB_POP2_1(ii,i) = sum(sum(triu(SUB_POP2_1_temp)))/(sum(sum(triu(DIST_BREED2t))));    
    
    end
    i
end
   

%% Concatenate arrays
SUB_POP = [SUB_POP_0;SUB_POP_1]; % concatenate first and second migration values in a unique array (Sub-population 1 + 2)
SUB_POP1 = [SUB_POP1_0;SUB_POP1_1]; % concatenate first and second migration values in a unique array (Sub-population 1)
SUB_POP2 = [SUB_POP2_0;SUB_POP2_1]; % concatenate first and second migration values in a unique array (Sub-population 2)
Sub_pop_BTW = [Sub_pop_BTW_0;Sub_pop_BTW_1]; % concatenate first and second migration values in a unique array ( Between Sub-population 1 and 2)

%% Calculating medians across different model experiments (in this example 5)
median_SUB_POP = zeros(1,length(Index)*2); % pre allocate array
median_SUB_POP_BTW = zeros(1,length(Index)*2); % pre allocate array
for k = 1:length(Index)*2
    median_SUB_POP(k) = median(SUB_POP(k,1:5));
    median_SUB_POP_BTW(k) = median(Sub_pop_BTW(k,1:5));
end

%% Plotting mean SR dissimilarity

figure(1) % define figure

ymax = max(max(max(SUB_POP,Sub_pop_BTW)))+1; % calculating the maximum x value for the plot
%% define the breeding and feeding areas  and their shading in the plot
v1 = [28 0; 28 ymax;61 ymax; 61 0; 149 0; 149 ymax;182 ymax; 182 0]; %breeding season shading
f1 = [1 2 3 4; 5 6 7 8]; %breeding season shading
P1 = patch('Faces',f1,'Vertices',v1,'FaceColor',[0.9,0.9,0.9],'FaceAlpha',.3, 'EdgeColor', 'none'); %breeding season shading
v2 = [91 0; 91 ymax;121 ymax; 121 0; 212 0;212 ymax;242 ymax; 242 0]; % feeding season shading
f2 = [1 2 3 4; 5 6 7 8]; % feeding season shading
P2 = patch('Faces',f2,'Vertices',v2,'FaceColor',[0.6,0.6,0.6],'FaceAlpha',.2, 'EdgeColor', 'none'); % feeding season shading
hold on

%% Set font size and x&y limits
set(gca,'FontSize',30,'Xtick',[]);
xlim([0 length(SUB_POP)]);
ylim([0 ymax]);
hold on

%% Plot the mean SR dissimilarity
X1 = plot(SUB_POP,'LineWidth',0.2,'Color',[0.25, 0.25, 0.9, 0.2]); % Mean SR dissimilarity of ALL modelling experiments within population 1 + 2
hold on
X2 = plot(median_SUB_POP,'LineWidth',4,'Color',[0.25, 0.25, 0.9]); % Median of mean SR dissimilarity within population 1 + 2
hold on
X3 = plot(median_SUB_POP_BTW,'LineWidth',4,'Color',[0.9100, 0.4100, 0.1700]); % Median of mean SR dissimilarity between population 1 & 2
hold on
X4 = plot(Sub_pop_BTW,'LineWidth',0.2,'Color',[0.9100, 0.4100, 0.1700, 0.2]); % Mean SR dissimilarity of ALL modelling experiments between population 1 & 2
hold on

%% Set the lines corresponding to mean SR dissimilarity for songs recorded in eastern Australia in 2002 and 2003
x5=[0,242];
y5=[3.6,3.6];
x6=[0,242];
y6=[2.4,2.4];
X5 = plot(x5,y5,'Color' ,'k','LineWidth',2, 'LineStyle', '- -'); % 2003
hold on
X6 = plot(x6,y6,'Color' ,'k','LineWidth',2, 'LineStyle', ':'); % 2002
hold on

%% Set the legent
legend([X2 X3 X5 X6 P1 P2],  'Within Population (Median)',  'Between Populations (Median)', '2003', '2002', 'Breeding Season', 'Feeding Season')
