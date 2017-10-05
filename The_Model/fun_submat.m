function [Index, X, Y, Song_length, Sing_state, SUB_d,SUB_POP, parameters] = fun_submat(filename,iteration)
%% Set the number of nw
load(filename);
parameters.nw = length(whalenumber_0.iter_300.IntFact);
%% to access an agent you need to write
iterNames = fieldnames(whalenumber_0);
iterNames = iterNames(~ismember(iterNames,'name'));
nStores = size(iterNames,1); %assumes always have whalenumber_0!

iterNum = zeros(nStores,1);

for i=1:size(iterNames,1)
    subInd = strfind(iterNames{i},'_');
    if subInd
        %iterNum(i) = str2num(iterNames{i}(strfind(iterNames{i},'_')+1:end));
        iterNum(i) = str2double(iterNames{i}(strfind(iterNames{i},'_')+1:end));
    end
end

iterNum = sort(iterNum);
iterNum

%% Create arrays before the loop
X = zeros(nStores,parameters.nw);
Y = X;
% ShE_mat = X;
% ShE_song = ShE_mat;
Song_length = X;
Sing_state = X;
SUB_d = zeros(parameters.nw,parameters.nw);
% SUB_POP = zeros(nStores,parameters.nw,parameters.nw);

%nStores = 10;
A= 1:nStores(end) ;
N=50;
Index=A(1:N:length(A));

%% LOOP
for  kk =  1 : nStores
    kk
    for ii = 1:parameters.nw
        for j = 1:parameters.nw
        xname = eval(['whalenumber_' num2str(ii-1) '.iter_' num2str(iterNum(kk))]);
        yname = eval(['whalenumber_' num2str(j-1) '.iter_' num2str(iterNum(kk))]);                    
        %% Location
        X(kk,ii) = xname.X;
        Y(kk,ii) = xname.Y;
        %% Convergence (entropy)
%         ShE_mat(kk,ii) = wentropy(xname.matrix,'shannon');
%         ShE_song(kk,ii) = wentropy(xname.song,'shannon');
        %% Song length
        Song_length(kk,ii) = length(xname.song);
        %% Sing state
        Sing_state(kk,ii) = xname.singStates;
        %% SUB_MAT
        % Define the matrices
        ii_MAT = xname.matrix;
        j_MAT = yname.matrix;
        % First sum of sums
        SUB_mat = abs(ii_MAT - j_MAT);
        SUB_d(ii,j) = sum(sum(SUB_mat));
       
        
        SUB_POP(kk) = sum(sum(triu(SUB_d)));
        
        
        
 
        end
    end
end



end
%% Saving
% ShE_mat_FGS50CF0_0 = ShE_mat;
% ShE_song_FGS50CF0_0 = ShE_song;
% Song_length_FGS50CF0_0 = Song_length;
% Sing_state_FGS50CF0_0 = Sing_state;
% Sub_pop_FGS50CF0_0 = SUB_d;
% Index = B;
% save('EX6_FGS50CF0_0' ,'Song_length_FGS50CF0_0','Sing_state_FGS50CF0_0','Sub_pop_FGS50CF0_0','Index');