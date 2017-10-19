% Script Author and Affiliation: Luca Lamoni, University of St Andrews
% Paper title: “Using agent-based models to understand the role of individuals in the song evolution of humpback whales ( _Megaptera novaeangliae_ )”
% Please provide comments and questions at ll42@st-andrews.ac.uk

% We use this script to run the extraction_function on multiple .mat
% files at the same time using a loop

%% Main loop
for i= 1:5 % 1 to total number of files present in the folder that were produced by the agent-based model (you might need to change this accroding to what you run)
    for ii = 1:2 % if we run multiple migrations we can run through the different files here (in this example, 2 migration cycles)
    filename = ['ExperimentN' num2str(i-1) 'migrationNumber' num2str(ii-1) '.mat']; % here we specify the filename that will be used in the extraction_function
    [Index, X, Y, Song_length, Sing_state, SUB_d, parameters] = extraction_function(filename); % here we run the actual fuction extraction_function. See extraction_function.m for details
    save(['Data_' num2str(i-1) '_Mig' num2str(ii-1)],'Index','X', 'Y','Song_length','Sing_state','SUB_d','parameters'); % here we save a .mat file using i and ii to maintain the file and migration numbering 
    end
end


%%

