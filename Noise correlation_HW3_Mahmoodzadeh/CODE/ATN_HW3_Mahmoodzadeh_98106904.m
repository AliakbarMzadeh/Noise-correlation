%% Advanced Topics in Neuroscience - Aliakbar Mahmoodzadeh 98106904
            %%%%%%% ========= HW3 ========= %%%%%%%


%% pre-processing 

clc; clear; close all;

% pre-process function of the paper


%%  load the  datas

clc; close all;

% monkey1
monkey1Data = load('S_monkey1.mat');
% monkey2
monkey2Data = load('S_monkey2.mat');
% monkey3
monkey3Data = load('S_monkey3.mat');

% load removed datas
removedNeurons = load('removed.mat');


%% part1 - Tuning Curve for monkey 1
clc; close all;

% initialization
grating = 0:30:330;

% monkey1
% find number of spikes averaged over all trials for each neuron
neuronNumbers = 83;
nspMonkey1 = zeros(neuronNumbers,12);
for i=1:size(nspMonkey1,2)
    nspMonkey1(:,i) = sum(monkey1Data.S(i).mean_FRs,2);
end

[targetNeuron, ~] = find(ismember(nspMonkey1,...
    max(nspMonkey1(:))));
figure;
subplot(2,3,1)
set(gca,'color','r')
plot(grating,nspMonkey1(targetNeuron,:),'LineWidth',2,'Color','r');
grid on; grid minor;
xlabel('Orientation','interpreter','latex','Color','b');
ylabel('firing rate(Hz)','interpreter','latex','Color','b');
title("Tuning Curve - Monkey 1 - Neuron NO. = " + targetNeuron,'interpreter','latex','Color','b');
legend('The most active neuron','LineWidth',2);
xlim([0 330]);


targetNeuron = randperm(neuronNumbers,5);
for i=2:6
    subplot(2,3,i);
    set(gca,'color','r')
    plot(grating,nspMonkey1(targetNeuron(i-1),:),'LineWidth',2,'Color','k');
    grid on; grid minor;
    xlabel('Grating Degree','interpreter','latex','Color','b');
    ylabel('firing rate(Hz)','interpreter','latex','Color','b');
    title("Tuning Curve - Monkey 1 - Neuron NO. = " + targetNeuron(i-1),'interpreter','latex','Color','b');
    legend('Tuning curve','LineWidth',1);
    xlim([0 330]);
end
    



clc; close all;


%grating = 0:30:330;


neuronNumbers = 59;
nspMonkey = zeros(neuronNumbers,12);
for i=1:size(nspMonkey,2)
    nspMonkey(:,i) = sum(monkey2Data.S(i).mean_FRs,2);
end

[targetNeuron, ~] = find(ismember(nspMonkey,...
    max(nspMonkey(:))));

figure;
subplot(1,3,1);
plot(grating,nspMonkey(targetNeuron,:),'LineWidth',2,'Color','r');
grid on; grid minor;
xlabel('Orientation','interpreter','latex','Color','b');
ylabel('firingRate(Hz)','interpreter','latex','Color','b');
title("Tuning Curve - Monkey 2 - Neuron No. = 45" ,'interpreter','latex','Color','b');
legend('The most active neuron','LineWidth',2);
xlim([0 330]);

targetNeuron = randperm(neuronNumbers,5);
for i=2:3
    subplot(1,3,i);
    plot(grating,nspMonkey(targetNeuron(i-1),:),'LineWidth',2,'Color','k');
    grid on; grid minor;
    xlabel('Orientation','interpreter','latex','Color','b');
    ylabel('firingRate(Hz)','interpreter','latex','Color','b');
    title("Tuning Curve Monkey 2 - Neuron No. = " + targetNeuron(i-1),'interpreter','latex','Color','b');
    legend('Tuning curve','LineWidth',1);
    xlim([0 330]);
end
    
%%




grating = 0:30:330;


neuronNumbers = 105;
nspMonkey3 = zeros(neuronNumbers,12);
for i=1:size(nspMonkey3,2)
    nspMonkey3(:,i) = sum(monkey3Data.S(i).mean_FRs,2);
end

[targetNeuron, ~] = find(ismember(nspMonkey3,...
    max(nspMonkey3(:))));

figure;
subplot(1,3,1)
plot(grating,nspMonkey3(targetNeuron,:),'LineWidth',2,'Color','r');
grid on; grid minor;
xlabel('Orientation','interpreter','latex','Color','b');
ylabel('firingRate(Hz)','interpreter','latex','Color','b');
title(" Monkey 3 - Neuron No. = 20" ,'interpreter','latex' ,'Color','b');
legend('The most active neuron','LineWidth',2);
xlim([0 330]);

targetNeuron = randperm(neuronNumbers,5);
for i=2:3
    subplot(1,3,i);
    plot(grating,nspMonkey3(targetNeuron(i-1),:),'LineWidth',2,'Color','k');
    grid on; grid minor;
    xlabel('Orientation','interpreter','latex','Color','b');
    ylabel('firingRate(Hz)','interpreter','latex','Color','b');
    title(" Monkey 3 - Neuron No. = " + targetNeuron(i-1),'interpreter','latex','Color','b');
    legend('Tuning curve','LineWidth',1);
    xlim([0 330]);
end
    
%%
%% part1 - Tuning Curve for monkeys 1, 2, and 3
clc; close all;



grating = 0:30:330;

% monkey1
% find number of spikes averaged over all trials for each neuron
neuronNumbers = 83;
nspMonkey1 = zeros(neuronNumbers,12);
for i=1:size(nspMonkey1,2)
    nspMonkey1(:,i) = sum(monkey1Data.S(i).mean_FRs,2);
end

% Calculate the average tuning curve for all neurons in monkey 1
avhtcMonkey1 = mean(nspMonkey1, 1);


% monkey-2

grating = 0:30:330;


% find number of spikes averaged over all trials for each neuron
neuronNumbers = 59;
nspMonkey = zeros(neuronNumbers,12);
for i=1:size(nspMonkey,2)
    nspMonkey(:,i) = sum(monkey2Data.S(i).mean_FRs,2);
end

% Calculate the average tuning curve for all neurons in monkey 1
avgtcMonkey2 = mean(nspMonkey, 1);



% monkey-3

grating = 0:30:330;


% find number of spikes averaged over all trials for each neuron
neuronNumbers = 105;
nspMonkey3 = zeros(neuronNumbers,12);
for i=1:size(nspMonkey3,2)
    nspMonkey3(:,i) = sum(monkey3Data.S(i).mean_FRs,2);
end

% Calculate the average tuning curve for all neurons in monkey 1
avhtcMonkey3 = mean(nspMonkey3, 1);




% Plot the average tuning curves for all neurons in monkeys 1, 2, and 3 in one plot
figure;
hold on;
plot(grating, avhtcMonkey1, 'LineWidth', 2);
plot(grating, avgtcMonkey2, 'LineWidth', 2);
plot(grating, avhtcMonkey3, 'LineWidth', 2);
hold off;
grid on; grid minor;
xlabel('Grating', 'interpreter', 'latex','Color','b');
ylabel('firing rate(Hz)', 'interpreter', 'latex','Color','b');
title("Average Tuning Curves for Monkeys 1, 2, and 3", 'interpreter', 'latex','Color','b');
xlim([0 330]);
legend('Monkey 1', 'Monkey 2', 'Monkey 3', 'interpreter', 'latex');

% Store the average tuning curves for all neurons in monkeys 1, 2, and 3 in a cell array
averageTuningCurves = {avhtcMonkey1, avgtcMonkey2, avhtcMonkey3};

% Plot the average tuning curves for all neurons in monkeys 1, 2, and 3 in separate subplots
figure;
for m = 1:3
    subplot(1, 3, m);
    plot(grating, averageTuningCurves{m}, 'LineWidth', 2,'Color','k');
    grid on; grid minor;
    xlabel('Grating', 'interpreter', 'latex','Color','b');
    ylabel('firing rate(Hz)', 'interpreter', 'latex','Color','b');
    title("Average Tuning Curve for Monkey " + m, 'interpreter', 'latex','Color','b');
    xlim([0 330]);
end

%% part2 
for k=1:3
    load(['S_monkey' num2str(k) '.mat']);
    max_vec = [];
    for i=1:12
        max_vec = [max_vec; sum(S(i).mean_FRs')]
    end

    max_vec = max(max_vec);
    [~, ind] = sort(max_vec);

    data = [];
    for i=1:12
        data = [data max(S(i).mean_FRs(ind(end),:))];
    end
    
    hold on
    
end

%% Part2
for k=1:3
    load(['data_monkey' num2str(k) '_gratings.mat']);
    indices = 1:150;
    load('removed.mat');
    indices = indices(sum([0;0;removed_neurons{k,1}]==1:length(indices))==0);
    indices = indices(sum([0;0;removed_neurons{k,2}]==1:length(indices))==0);
    
    Map = data.MAP;
    Channels = data.CHANNELS;
    data_mesh = zeros(10,10);
    
    load(['S_monkey' num2str(k) '.mat']);
    
    for i=1:10
        for j=1:10
            [r, c] = find(Channels(:,1)==Map(i,j));
            [c, r] = find(indices == r);
            data_tune = [];
            for q=1:12
                data_tune = [data_tune max(max(S(q).mean_FRs(r,:)))];
            end
            [v, ind] = max(data_tune);
            if(size(ind,1)==0)
                data_mesh(i,j) = nan;
            else
                data_mesh(i,j) = ind;
            end
        end
    end
    figure;
    set(gcf,'Color',[1 1 1]);
    
    h = heatmap((data_mesh-1)*30);
    hmap(1:256,1) = linspace(0,1,256);
    hmap(:,[2 3]) = 0.7;
    huemap = hsv2rgb(hmap);
    colormap(huemap)
    title(['Monkey ' num2str(k)]);
    
    % Remove the numbers from the heatmap
    h.CellLabelColor = 'none';
    
    % Change the colormap to blue
    h.Colormap = parula;
    
   
end

%% part 2
for k=1:3
    load(['data_monkey' num2str(k) '_gratings.mat']);
    indices = 1:150;
    load('removed.mat');
    indices = indices(sum([0;0;removed_neurons{k,1}]==1:length(indices))==0);
    indices = indices(sum([0;0;removed_neurons{k,2}]==1:length(indices))==0);
    
    Map = data.MAP;
    Channels = data.CHANNELS;
    data_mesh = zeros(10,10);
    
    load(['S_monkey' num2str(k) '.mat']);
    
    for i=1:10
        for j=1:10
            [r, c] = find(Channels(:,1)==Map(i,j));
            [c, r] = find(indices == r);
            data_tune = [];
            for q=1:12
                data_tune = [data_tune max(max(S(q).mean_FRs(r,:)))];
            end
            [v, ind] = max(data_tune);
            if(size(ind,1)==0)
                data_mesh(i,j) = nan;
            else
                data_mesh(i,j) = ind;
            end
        end
    end

    figure;
    set(gcf, 'Color', [1 1 1]);
    

    % Create a custom heatmap using imagesc
    h = imagesc((data_mesh - 1) * 30);
    colormap(parula); % Set the colormap to blue
    colorbar;
    axis equal tight;

    % Set the title with the LaTeX interpreter
    title(['Monkey ' num2str(k)], 'interpreter', 'latex');

    % Set the x-axis and y-axis labels with the LaTeX interpreter
    xlabel('X Axis Label', 'interpreter', 'latex');
    ylabel('Y Axis Label', 'interpreter', 'latex');

    % Set the x-axis and y-axis tick labels with the LaTeX interpreter
    set(gca, 'XTick', 1:10, 'XTickLabel', arrayfun(@num2str, 1:10, 'UniformOutput', false), 'TickLabelInterpreter', 'latex');
    set(gca, 'YTick', 1:10, 'YTickLabel', arrayfun(@num2str, 1:10, 'UniformOutput', false), 'TickLabelInterpreter', 'latex');

   
end
%% part 3 

%% part 3

for k=1:3
    r_sig = [];
    r_noise = [];
    dis = [];
    load(['S_monkey' num2str(k) '.mat']);
    for i=1:size(S(1).mean_FRs,1)
        datai = [];
        for i0=1:12
            datai = [datai sum(S(i0).mean_FRs(i,:))];
        end
        for j=1:i
            dataj = [];
            for i0=1:12
                dataj = [dataj sum(S(i0).mean_FRs(j,:))];
            end
            r = corrcoef(datai,dataj);
            r_sig(i,j) = r(1,2);
            r_sig(j,i) = r(1,2);
        end
    end
    for i=1:size(S(1).mean_FRs,1)
        datai = [];
        for q=1:12
            t = [];
            for t0=1:200
                t = [t; sum(S(q).trial(t0).counts(i,:))];
            end
            datai = [datai; zscore(t)];
        end
        for j=1:i
            dataj = [];
            for q=1:12
                t = [];
                for t0=1:200
                    t = [t; sum(S(q).trial(t0).counts(j,:))];
                end
                dataj = [dataj; zscore(t)];
            end
            r = corrcoef(datai,dataj);
            r_noise(i,j) = r(1,2);
            r_noise(j,i) = r(1,2);
        end
    end
    load(['data_monkey' num2str(k) '_gratings.mat']);
    Map = data.MAP;
    Channels = data.CHANNELS;
    indices = 1:150;
    load('removed.mat');
    indices = indices(sum([0;0;removed_neurons{k,1}]==1:length(indices))==0);
    indices = indices(sum([0;0;removed_neurons{k,2}]==1:length(indices))==0);
    for i=1:size(S(1).mean_FRs,1)
        [xi, yi] = find(Channels(indices(i),1) == Map);
        for j=1:i
            [xj, yj] = find(Channels(indices(j),1) == Map);
            dis(i,j) = sqrt((0.4*xi-0.4*xj)^2+(0.4*yi-0.4*yj)^2);
            dis(j,i) = sqrt((0.4*xi-0.4*xj)^2+(0.4*yi-0.4*yj)^2);
        end
    end
    r_sig = r_sig(tril(ones(size(r_sig)),-1)==1);
    r_noise = r_noise(tril(ones(size(r_noise)),-1)==1);
    dis = dis(tril(ones(size(dis)),-1)==1);
    
    
    G1 = r_sig > 0.5;
    G2 = r_sig > 0 & r_sig <= 0.5;
    G3 = r_sig > -0.5 & r_sig <=0;
    G4 = r_sig <= -0.5;
    figure;
    set(gcf,'Color',[1 1 1]);
    
    hold on
    
    [Y,E] = discretize(dis(G1),8);
    d = [];
    err = [];
    t = r_noise(G1);
    for i=1:9
        d = [d mean(t(Y==i))]
        err = [err std(t(Y==i))/sqrt(length(t(Y==i)))];
    end
    errorbar(E,d,err)
    
    [Y,E] = discretize(dis(G2),8);
    d = [];
    err = [];
    t = r_noise(G2);
    for i=1:9
        d = [d mean(t(Y==i))]
        err = [err std(t(Y==i))/sqrt(length(t(Y==i)))];
    end
    errorbar(E,d,err)
    
    [Y,E] = discretize(dis(G3),8);
    d = [];
    err = [];
    t = r_noise(G3);
    for i=1:9
        d = [d mean(t(Y==i))]
        err = [err std(t(Y==i))/sqrt(length(t(Y==i)))];
    end
    errorbar(E,d,err)
    
    [Y,E] = discretize(dis(G4),8);
    d = [];
    err = [];
    t = r_noise(G4);
    for i=1:9
        d = [d mean(t(Y==i))]
        err = [err std(t(Y==i))/sqrt(length(t(Y==i)))];
    end
 
    
    % Second Plot
    G1 = dis <= 1;
    G2 = dis > 1 & dis <=2;
    G3 = dis > 2 & dis <=3;
    G4 = dis > 3 & dis <=4;
    G5 = dis > 5 & dis <=10;
    figure;
    set(gcf,'Color',[1 1 1]);
    
    hold on
    
    [Y,E] = discretize(r_sig(G1),8);
    d = [];
    err = [];
    t = r_noise(G1);
    for i=1:9
        d = [d mean(t(Y==i))]
        err = [err std(t(Y==i))/sqrt(length(t(Y==i)))];
    end
    errorbar(E,d,err)
    
    [Y,E] = discretize(r_sig(G2),8);
    d = [];
    err = [];
    t = r_noise(G2);
    for i=1:9
        d = [d mean(t(Y==i))]
        err = [err std(t(Y==i))/sqrt(length(t(Y==i)))];
    end
    errorbar(E,d,err)
    
    [Y,E] = discretize(r_sig(G3),8);
    d = [];
    err = [];
    t = r_noise(G3);
    for i=1:9
        d = [d mean(t(Y==i))]
        err = [err std(t(Y==i))/sqrt(length(t(Y==i)))];
    end
    errorbar(E,d,err)
    
    [Y,E] = discretize(r_sig(G4),8);
    d = [];
    err = [];
    t = r_noise(G4);
    for i=1:9
        d = [d mean(t(Y==i))]
        err = [err std(t(Y==i))/sqrt(length(t(Y==i)))];
    end
   
    

    
 
    rs = linspace(-1,1,11);
    ds = linspace(0,4,9);
    data_heat = [];
    for i=2:length(rs)
        for j=2:length(ds)
            sel = (r_sig>rs(i-1) & r_sig<=rs(i)) .* (dis>ds(j-1) & dis<=ds(j));
            data_heat(i-1,j-1) = mean(r_noise(sel==1));
        end
    end
% figure;
% set(gcf,'Color',[1 1 1]);

% imagesc(data_heat);
% colormap(jet); % You can choose your desired colormap here
% colorbar;
% xticks(1:9);
% xticklabels(0:0.5:4);
% yticks(1:11);
% yticklabels(-1:0.2:1);
% xlabel('Distance between electrodes (mm)','interpreter','latex');
% ylabel('Spike count correlation (rsc)','interpreter','latex');
% title(['Monkey ' num2str(k)]);


figure;
set(gcf,'Color',[1 1 1]);
h = heatmap(data_heat);
h.Colormap = parula; % You can choose your desired colormap here
x_labels = linspace(0, 4, size(data_heat, 2));
y_labels = linspace(-1, 1, size(data_heat, 1));
h.XDisplayLabels = x_labels;
h.YDisplayLabels = y_labels;
xlabel('Distance between electrodes (mm)','interpreter','latex');
ylabel('Spike count correlation (rsc)','interpreter','latex');
title(['Monkey ' num2str(k)]);


end

%% functions

function spikecounts = bin_spikes(spikes, binWidth)
% S = bin_spikes(spikes, binWdith)
% 
% bins the spikes with given binWidth (in ms)
%
% INPUT:
%   spikes (num_neurons x num_1ms_timepoints):  contains binary
%       spike information in 1ms bins
%   binWidth (in ms): bins spikes in non-overlapping windows with
%       specified bin width
%
% OUTPUT:
%   spikecounts (num_neurons x num_timebins): contains
%       binned spike counts
%
% NOTES:
%   If a trial has a number of timepoints not evenly divisible by
%       the binWidth, the remainder timepoints are removed and not
%       considered during the spiking.
%
%
% Author: bcowley, 2014


    spikecounts = [];
    
    num_timepoints = size(spikes,2);
    
    if (num_timepoints < binWidth)
        warning('bin_spikes:  Number of timepoints is less than binWidth.');
    end
            
    bin_indices = 1:binWidth:num_timepoints;

    if (mod(num_timepoints,binWidth) ~= 0 && length(bin_indices) > 1) % the last bin does not have enough time points in it
        bin_indices = bin_indices(1:(end-1));
    end

    for ibin = 1:length(bin_indices)
        
        spikecounts = [spikecounts ...
            sum(spikes(:, bin_indices(ibin):(bin_indices(ibin)+binWidth -1)),2)];
    end            
 

end


 function [S] = computeSpontCounts(EVENTS,bininseconds,spacebetweenbins)
% 
% [S, channels, medSNR] = computeSpontCounts(EVENTS,bininseconds,chunksize)
%
% Computes spike counts from spiking event times. 
%
% Inputs: 
%     EVENTS:           Cell array of spiking events for each sorted unit
%     bininseconds:     Number of seconds over which spike counts will be
%                       binned
%     spacebetweenbins: Time between successive bins.
%
% Outputs:
%     S:                N X T matrix of spike counts with N neurons and T 
%                       trials. Adjacent columns represent spike counts for
%                       adjacent time bins of size bininseconds
% Author: Ryan Williamson, Oct. 2016
%

 
maxtime = max(cell2mat(cellfun(@(x)max(x),EVENTS,'UniformOutput',false)));
maxtrials = floor(maxtime/(bininseconds+spacebetweenbins));
S =zeros(length(EVENTS), maxtrials);
for n = 1:maxtrials
    starttrial = (n-1)*(bininseconds+spacebetweenbins);
    endtrial = starttrial+bininseconds;
    S(:,n) = cell2mat(cellfun(@(x)sum(x>=starttrial&x<endtrial),EVENTS,'UniformOutput',false))';
end

 end

 %%
 %  This script contains three parts:
%   1. Convert spike times to 1ms bins.
%   2. Remove bad/very low firing units.
%   3. Compute the trial-averaged population activity (PSTHs).
%
%  S struct is used to store the data:
%    S(igrat).trial(itrial).spikes  (num_units x num_1ms_timebins)
%    S(igrat).trial(itrial).counts  (num_units x num_20ms_timebins)
%    S(igrat).mean_FRs  (num_units x num_20ms_timebins)
%
%  Author: Ben Cowley, bcowley@cs.cmu.edu, Oct. 2016
%
%  Notes:
%    - automatically saves 'S' in ./spikes_gratings/


%% parameters

    SNR_threshold = 1.5;
    firing_rate_threshold = 1.0;  % 1.0 spikes/sec
    binWidth = 20;  % 20 ms bin width

    
%% parameters relevant to experiment

    length_of_gratings = 1;  % each gratings was shown for 1.28s, take the last 1s
    
    filenames{1} = './spikes_gratings/data_monkey1_gratings.mat';
    filenames{2} = './spikes_gratings/data_monkey2_gratings.mat';
    filenames{3} = './spikes_gratings/data_monkey3_gratings.mat';


    monkeys = {'monkey1', 'monkey2', 'monkey3'};
    

%%  spike times --> 1ms bins

    for imonkey = 1:length(monkeys)
        S = [];

        fprintf('binning spikes for %s\n', monkeys{imonkey});

        load(filenames{imonkey});
            % returns data.EVENTS

        num_neurons = size(data.EVENTS,1);
        num_gratings = size(data.EVENTS,2);
        num_trials = size(data.EVENTS,3);

        edges = 0.28:0.001:1.28;  % take 1ms bins from 0.28s to 1.28s

        for igrat = 1:num_gratings
            for itrial = 1:num_trials
                for ineuron = 1:num_neurons
                    S(igrat).trial(itrial).spikes(ineuron,:) = histc(data.EVENTS{ineuron, igrat, itrial}, edges);
                end
                S(igrat).trial(itrial).spikes = S(igrat).trial(itrial).spikes(:,1:end-1);  % remove extraneous bin at the end
            end
        end

        save(sprintf('./spikes_gratings/S_%s.mat', monkeys{imonkey}), 'S', '-v7.3');
    end
    




%%  Pre-processing:  Remove bad/very low firing units

    % remove units based on low SNR
    
    for imonkey = 1:length(monkeys)
        load(filenames{imonkey});
            % returns data.SNR
        keepNeurons = data.SNR >= SNR_threshold;
        clear data;
        
        fprintf('keeping units with SNRs >= %f for %s\n', SNR_threshold, monkeys{imonkey});
        
        load(sprintf('./spikes_gratings/S_%s.mat', monkeys{imonkey}));
            % returns S(igrat).trial(itrial).spikes
        
        num_grats = length(S);
        num_trials = length(S(1).trial);
        
        for igrat = 1:num_grats
            for itrial = 1:num_trials
                S(igrat).trial(itrial).spikes = S(igrat).trial(itrial).spikes(keepNeurons,:);
            end
        end
        
        save(sprintf('./spikes_gratings/S_%s.mat', monkeys{imonkey}), 'S', '-v7.3');
    end
    
    % remove units with mean firing rates < 1.0 spikes/sec
    
    for imonkey = 1:length(monkeys)
        load(sprintf('./spikes_gratings/S_%s.mat', monkeys{imonkey}));
            % returns S(igrat).trial(itrial).spikes
        num_grats = length(S);
        num_trials = length(S(1).trial);
        
        mean_FRs = [];   
        
        for igrat = 1:num_grats
            for itrial = 1:num_trials
                mean_FRs = [mean_FRs sum(S(igrat).trial(itrial).spikes,2)/1.0];
            end
        end
        
        mean_FRs_gratings = mean(mean_FRs,2);
        keepNeurons = mean_FRs_gratings >= firing_rate_threshold;
        
        for igrat = 1:num_grats
            for itrial = 1:num_trials
                S(igrat).trial(itrial).spikes = S(igrat).trial(itrial).spikes(keepNeurons,:);
            end
        end
           
        save(sprintf('./spikes_gratings/S_%s.mat', monkeys{imonkey}), 'S', '-v7.3');
        
    end


%%  Take spike counts in bins
    for imonkey = 1:length(monkeys)
        
        fprintf('spike counts in %dms bins for %s\n', binWidth, monkeys{imonkey});
        
        load(sprintf('./spikes_gratings/S_%s.mat', monkeys{imonkey}));
            % returns S(igrat).trial(itrial).spikes
        num_grats = length(S);
        num_trials = length(S(1).trial);
        
        for igrat = 1:num_grats
            for itrial = 1:num_trials
                S(igrat).trial(itrial).counts = bin_spikes(S(igrat).trial(itrial).spikes, binWidth);
            end
        end
        
        save(sprintf('./spikes_gratings/S_%s.mat', monkeys{imonkey}), 'S', '-v7.3');
    end
    
    
%%  Compute trial-averaged population activity (PSTHs)

    for imonkey = 1:length(monkeys)
        fprintf('computing PSTHs for %s\n', monkeys{imonkey});
        
        load(sprintf('./spikes_gratings/S_%s.mat', monkeys{imonkey}));
            % returns S(igrat).trial(itrial).spikes
        num_grats = length(S);
        num_trials = length(S(1).trial);
        
        for igrat = 1:num_grats
            mean_FRs = zeros(size(S(igrat).trial(1).counts));
            for itrial = 1:num_trials
                mean_FRs = mean_FRs + S(igrat).trial(itrial).counts;
            end
            S(igrat).mean_FRs = mean_FRs / num_trials;
        end
        
        save(sprintf('./spikes_gratings/S_%s.mat', monkeys{imonkey}), 'S', '-v7.3');
    end
    
%%
%  This script contains three parts:
%   1. Convert spike times to 1ms bins.
%   2. Remove bad/very low firing units.
%   3. Compute the trial-averaged population activity (PSTHs).
%
%  S and mean_FRs are used to store the data:
%    S(itrial).spikes  (num_units x num_1ms_timebins)
%    S(itrial).counts  (num_units x num_20ms_timebins)
%    mean_FRs (num_units x num_20ms_timebins)
%   
%  Author: Ben Cowley, bcowley@cs.cmu.edu, Oct. 2016
%
% Notes:
%   - files were spike sorted together
%   - automatically saves 'S' and 'mean_FRs' in ./spikes_movies/

%% parameters

    SNR_threshold = 2.0;
    firing_rate_threshold = 1.0;  % 1.0 spikes/sec
    binWidth = 20;  % in ms


%% parameters relevant to experiment

    length_of_movie = 30;  % each movie was 30 seconds
    
    filenames{1,1} = './spikes_movies/data_monkey1_gratings_movie.mat';
    filenames{1,2} = './spikes_movies/data_monkey1_natural_movie.mat';
    filenames{1,3} = './spikes_movies/data_monkey1_noise_movie.mat';
    filenames{2,1} = './spikes_movies/data_monkey2_gratings_movie.mat';
    filenames{2,2} = './spikes_movies/data_monkey2_natural_movie.mat';
    filenames{2,3} = './spikes_movies/data_monkey2_noise_movie.mat';

    monkeys = {'monkey1', 'monkey2'};
    movies = {'gratings_movie', 'natural_movie', 'noise_movie'};
    
    
%% match channels and units across movies

    for imonkey = 1:length(monkeys)
        channels = [];
        for imovie = 1:length(movies)
            load(filenames{imonkey, imovie});
                % returns data.CHANNELS
                
            channels{imovie} = [data.CHANNELS(:,1) + 1000 * data.CHANNELS(:,2)];
        end
        
        intersect_channels = intersect(channels{1}, channels{2});
        intersect_channels = intersect(intersect_channels, channels{3});
        
        for imovie = 1:length(movies)
            load(filenames{imonkey, imovie});
                % returns data.CHANNELS
                
            channels = [data.CHANNELS(:,1) + 1000 * data.CHANNELS(:,2)];
            keepChannels = ismember(channels, intersect_channels);
            
            data.EVENTS = data.EVENTS(keepChannels,:);
            data.CHANNELS = data.CHANNELS(keepChannels,:);
            data.SNR = data.SNR(keepChannels);
            
            save(filenames{imonkey, imovie}, 'data', '-v7.3');
        end
        
    end
   

%%  spike times --> 1ms bins

    for imonkey = 1:length(monkeys)
        for imovie = 1:length(movies)
            S = [];
            
            fprintf('binning spikes for %s %s\n', monkeys{imonkey}, movies{imovie});
            
            load(filenames{imonkey, imovie});
                % returns data.EVENTS
                
            num_neurons = size(data.EVENTS,1);
            num_trials = size(data.EVENTS,2);
            
            edges = 0:0.001:30;  % take 1ms bins
            
            for itrial = 1:num_trials
                for ineuron = 1:num_neurons
                    S(itrial).spikes(ineuron,:) = histc(data.EVENTS{ineuron, itrial}, edges);
                end
                S(itrial).spikes = S(itrial).spikes(:,1:end-1);  % remove extraneous bin at the end
            end
                
            save(sprintf('./spikes_movies/S_%s_%s.mat', monkeys{imonkey}, movies{imovie}), 'S', '-v7.3');
        end
    end
    




%%  Pre-processing:  Remove bad/very low firing units

    % remove units based on SNR <= SNR_threshold
    
    for imonkey = 1:length(monkeys)
        keepNeurons = [];
        for imovie = 1:length(movies)
            load(filenames{imonkey, imovie});
                % returns data.SNR
            if (isempty(keepNeurons))
                keepNeurons = data.SNR >= SNR_threshold;
            else
                keepNeurons = keepNeurons & data.SNR >= SNR_threshold;  % SNRs are different across movies
            end
        end
        clear data;
        
        for imovie = 1:length(movies)
            S = [];
            
            fprintf('keeping units with SNRs >= %f for %s %s\n', SNR_threshold, monkeys{imonkey}, movies{imovie});

            load(sprintf('./spikes_movies/S_%s_%s.mat', monkeys{imonkey}, movies{imovie}));
                % returns S(itrial).spikes
                
            num_trials = length(S);

            for itrial = 1:num_trials
                S(itrial).spikes = S(itrial).spikes(keepNeurons,:);  
            end
                
            save(sprintf('./spikes_movies/S_%s_%s.mat', monkeys{imonkey}, movies{imovie}), 'S', '-v7.3');
        end
    end
    
    % remove units with mean firing rates < firing rate threshold
    
    for imonkey = 1:length(monkeys)
        mean_FRs_movies = [];
        for imovie = 1:length(movies)
            S = [];
            
            load(sprintf('./spikes_movies/S_%s_%s.mat', monkeys{imonkey}, movies{imovie}));
                % returns S(itrial).spikes
                
            num_trials = length(S);
            mean_FRs = [];
            for itrial = 1:num_trials
                mean_FRs(:,itrial) = sum(S(itrial).spikes,2)/30.0;  % divide by length of movie
            end
            
            mean_FRs_movies = [mean_FRs_movies mean_FRs];
        end
        
        mean_FRs_movies = mean(mean_FRs_movies,2);
        
        keepNeurons = mean_FRs_movies >= firing_rate_threshold;
        
        for imovie = 1:length(movies)
            fprintf('keeping units with mean firing rates >= %d spikes/sec for %s %s\n', firing_rate_threshold, monkeys{imonkey}, movies{imovie});

            load(sprintf('./spikes_movies/S_%s_%s.mat', monkeys{imonkey}, movies{imovie}));
                % returns S(itrial).spikes

            num_trials = length(S);

            for itrial = 1:num_trials
                S(itrial).spikes = S(itrial).spikes(keepNeurons,:);
            end

            save(sprintf('./spikes_movies/S_%s_%s.mat', monkeys{imonkey}, movies{imovie}), 'S', '-v7.3');
        end

    end

    
%%  Take spike counts in 20ms bins
    for imonkey = 1:length(monkeys)
        for imovie = 1:length(movies)
            fprintf('spike counts in 20ms bins for %s %s\n', monkeys{imonkey}, movies{imovie});

            load(sprintf('./spikes_movies/S_%s_%s.mat', monkeys{imonkey}, movies{imovie}));
                % returns S(itrial).spikes
                
            num_trials = length(S);
            for itrial = 1:num_trials
                S(itrial).counts = bin_spikes(S(itrial).spikes, 20);
            end
            
            save(sprintf('./spikes_movies/S_%s_%s.mat', monkeys{imonkey}, movies{imovie}), 'S', '-v7.3');
        end
    end
    
%%  Compute trial-averaged population activity (PSTHs)

    for imonkey = 1:length(monkeys)
        for imovie = 1:length(movies)
            fprintf('computing PSTHs for %s %s\n', monkeys{imonkey}, movies{imovie});

            load(sprintf('./spikes_movies/S_%s_%s.mat', monkeys{imonkey}, movies{imovie}));
                % returns S(itrial).counts
                
            mean_FRs = zeros(size(S(1).counts));
            num_trials = length(S);
            for itrial = 1:num_trials
                mean_FRs = mean_FRs + S(itrial).counts;
            end
            mean_FRs = mean_FRs / num_trials;
            
            save(sprintf('./spikes_movies/mean_FRs_%s_%s.mat', ...
                monkeys{imonkey}, movies{imovie}), 'mean_FRs', '-v7.3');
        end
    end
        


        





