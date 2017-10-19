% Script Author and Affiliation: Luca Lamoni, University of St Andrews
% Paper title: “Using agent-based models to understand the role of individuals in the song evolution of humpback whales (Megaptera novaeangliae)”
% Please provide comments and questions at ll42@st-andrews.ac.uk

% We use the extraction_function to extract the data recorded in the .mat
% file produced by the model in the python code Unified_model.py

%Data available:
% Index: the index of model iterations we want to examine
% X & Y: geographical coordinates of each agent at each iteration
% Song_length: Song length of each song at each iteration
% Sing_state: Sing state of each agent at each iteration. 0 = No Singing, 
%1 = Singing
%SUB_d: Pairwise SR subtraction algorithm (see manuscript for details) 
% parameters: model parameter settings


function [Index, X, Y, Song_length, Sing_state, SUB_d, parameters] = extraction_function(filename)
%% Load filename and set the number of agents (nw)
load(filename);
parameters.nw = length(whalenumber_0.iter_0.IntFact);

%% Coding routine to assign agent names and iteration numbers 
iterNames = fieldnames(whalenumber_0);
iterNames = iterNames(~ismember(iterNames,'name'));
nStores = size(iterNames,1); %assumes always have whalenumber_0!

iterNum = zeros(nStores,1);
for i=1:size(iterNames,1)
    subInd = strfind(iterNames{i},'_');
    if subInd
        iterNum(i) = str2double(iterNames{i}(strfind(iterNames{i},'_')+1:end));
    end
end
iterNum = sort(iterNum);

%% Create empty arrays before the loop to store data and improove speed
X = zeros(nStores,parameters.nw);
Y = X;
Song_length = X;
Sing_state = X;
SUB_d = zeros(nStores,parameters.nw,parameters.nw);
%% This section of code might be used to subsample the iterations we want to examine
A= 1:nStores(end);
N=1; % Change N if you want to subsample iterations to analyse. Example: look only at 1 iteration every 10, then N = 10. 
Index=A(1:N:length(A));
 
%% Main loop that goes to every iteration and every agent to extract/calculate the parameters we are interested in.
for  kk =  Index
    for ii = 1:parameters.nw
        for j = 1:parameters.nw %we use both ii and j because some of the operations we are about to calculate are pairwise, so we need to define two agents
        %% We use eval to create the string to access the record of each agent at each iteration
        xname = eval(['whalenumber_' num2str(ii-1) '.iter_' num2str(iterNum(kk))]); %we use eval to create the string of each agent and iteration
        yname = eval(['whalenumber_' num2str(j-1) '.iter_' num2str(iterNum(kk))]);  
        
        %% Here we store in the vectors X and Y the coordinates of each agent in the population        
        X(kk,ii) = xname.X;
        Y(kk,ii) = xname.Y;
        
        %% Here we calculate for each agent at each iteration the length of its song
        Song_length(kk,ii) = length(xname.song);
        
        %% Here we simply store each agent's sing state at each iteration
        Sing_state(kk,ii) = xname.singStates;
        
        %% Pairwise SR subtraction method
        % First we define the two agents' SRs
        ii_MAT = xname.matrix;
        j_MAT = yname.matrix;
        % Secondly we make a subtraction the two SRs, note the use of abs before the
        % subtraction takes place
        SUB_mat = abs(double(ii_MAT) - double(j_MAT));
        % Finally, we store in a tridimensional matrix the sum (of the sum)
        % of the matrix resulting from the previous step
        SUB_d(kk,ii,j) = sum(sum(SUB_mat));
           
        end
    end
    kk % this is just used to have a visual idea of wich iteration is currently being processed
end
end
