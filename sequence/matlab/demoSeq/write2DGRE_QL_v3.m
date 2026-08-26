clear; close all;clc;

fn='2d_gre_100sli_5_FA_Nx_100_v3_invivo';

%% sequence system settings
sys = mr.opts('MaxGrad',60,'GradUnit','mT/m',...
    'MaxSlew',50,'SlewUnit','T/m/s',...
    'rfRingdownTime', 10e-6, 'rfDeadtime', 100e-6, 'B0', 2.89);

seq = mr.Sequence(sys);
fov = 200e-3;
Nx = 100;
Ny = 100;
dwell = 2e-6;                       % ADC sample time (s)
sliceThickness = 2e-3;              % slice thickness (m)
alpha = 77.107;                          % flip angle (degrees) 52.6 for phantom , 77.107 for brain
TR = 15e-3;                         % repetition time TR (s)
rfSpoilingInc = 117;                % RF spoiling increment
Nslices = 100;
t_pre = 2e-3; % duration of x pre-phaser
rotation_angles=[0:14]*pi/15;

% Create alpha-degree slice selection pulse and gradient
[rf, gz] = mr.makeSincPulse(alpha*pi/180, 'Duration', 3e-3, ...
    'SliceThickness', sliceThickness, 'apodization', 0.42, ...
    'use', 'excitation', ...
    'timeBwProduct', 4, 'system', sys);
gzReph = mr.makeTrapezoid('z', 'Area', -gz.area/2, 'Duration', t_pre, 'system', sys);

% Define other gradients and ADC events
deltak = 1/fov;
gx = mr.makeTrapezoid('x', 'FlatArea', Nx*deltak, 'FlatTime', Nx*dwell, 'system', sys);
adc = mr.makeAdc(Nx, 'Duration', gx.flatTime, 'Delay', gx.riseTime, 'system', sys);
gxPre = mr.makeTrapezoid('x', 'Area', -gx.area/2, 'Duration', t_pre, 'system',sys);
phaseAreas = ((0:Ny-1)-Ny/2)*deltak;
gyPre = mr.makeTrapezoid('y', 'Area', max(abs(phaseAreas)), ...
    'Duration', mr.calcDuration(gxPre), 'system', sys);
peScales = phaseAreas/gyPre.area;
gxSpoil = mr.makeTrapezoid('x', 'Area', 2*Nx*deltak, 'system', sys);
gzSpoil = mr.makeTrapezoid('z', 'Area', 4/sliceThickness, 'system', sys);
delayTR=ceil((TR - (mr.calcDuration(rf,gz)+ mr.calcDuration(gxPre, gyPre, gzReph)+  2*mr.calcDuration(gx)+mr.calcDuration(gxSpoil, gyPre, gzSpoil) ))/seq.gradRasterTime)*seq.gradRasterTime;

%%  interleaved slices
K = Nslices / 2;
freqOffset_factor = (0:Nslices-1) - (Nslices-1)/2;
interleaved_slic_indexS = reshape([1:K; K+1:Nslices], 1, []);
interleaved_freqOffset_factor = freqOffset_factor(interleaved_slic_indexS);
nDummyShots = 20;  % shots to reach steady state
pislquant = 10;     % number of shots/ADC events used for receive gain calibration

%% TR per iY (nominal TR = 15 ms per slice period)
% delayTR was set using unscaled gyPre; PE scaling pesc(iY) can change gradient durations slightly.
% TR_one_slice(iY): time for one RF + readout + spoil + delay for one slice at this iY.
% TR_all_slices(iY) = Nslices * TR_one_slice(iY): time to play all slices for one iY.
%
% Dummy iY (iY <= -pislquant) and receive-gain iY still use the same inner loop iSlice=1:Nslices,
% so you still get Nslices repetitions per iY — total time is still ~ Nslices*TR, not a shorter
% "dummy TR". Only iSlice==1 during dummy uses gx without adc on the first readout block; duration
% matches gx+adc if the ADC fits in the readout lobe (usual case).
iY_list = (-nDummyShots-pislquant+1):Ny;
TR_one_slice = zeros(size(iY_list));
for k = 1:length(iY_list)
    iY = iY_list(k);
    pesc = (iY>0) * peScales(max(iY,1));
    pesc = pesc + (pesc == 0)*eps;
    gy_s  = mr.scaleGrad(gyPre, pesc);
    gy_sn = mr.scaleGrad(gyPre, -pesc);
    TR_one_slice(k) = mr.calcDuration(rf,gz) + mr.calcDuration(gxPre, gy_s, gzReph) + ...
        2*mr.calcDuration(gx) + mr.calcDuration(gxSpoil, gy_sn, gzSpoil) + delayTR;
end
TR_all_slices = Nslices * TR_one_slice;
fprintf('Target TR (per slice) = %.6f ms\n', TR*1e3);
fprintf('  Actual TR_one_slice: min=%.6f ms, max=%.6f ms, mean=%.6f ms\n', ...
    min(TR_one_slice)*1e3, max(TR_one_slice)*1e3, mean(TR_one_slice)*1e3);
fprintf('  Time per iY (all %d slices): min=%.3f s, max=%.3f s, mean=%.3f s\n', ...
    Nslices, min(TR_all_slices), max(TR_all_slices), mean(TR_all_slices));

%% rotating view acquisition
for iview=[1]
    disp(['generate seq for view:', num2str(iview)])
    alpha_rad=rotation_angles(iview);
    rot_mat = [cos(alpha_rad), 0, sin(alpha_rad);0, 1, 0;-sin(alpha_rad), 0, cos(alpha_rad)];
    rot = mr.makeRotation(rot_mat);  % rotation event

    for iY = (-nDummyShots-pislquant+1):Ny

        rf_phase = 0;
        rf_inc = 0;

        for iSlice=1:Nslices
            isDummyTR = iY <= -pislquant;
            isReceiveGainCalibrationTR = iY < 1 & iY > -pislquant;
            % RF spoiling
            rf.phaseOffset = rf_phase/180*pi;
            adc.phaseOffset = rf_phase/180*pi;
            rf_inc = mod(rf_inc+rfSpoilingInc, 360.0);
            rf_phase = mod(rf_phase+rf_inc, 360.0);

            rf.freqOffset=gz.amplitude*sliceThickness*interleaved_freqOffset_factor(iSlice); % QL
            seq.addBlock(rf, gz); % , mr.makeLabel('SET', 'TRID', 1));

            % Slice-select refocus and readout prephasing
            % Set phase-encode gradients to zero while iY < 1
            pesc = (iY>0) * peScales(max(iY,1));  % phase-encode gradient scaling
            pesc = pesc + (pesc == 0)*eps;        % non-zero scaling so that the trapezoid shape is preserved in the .seq file

            seq.addBlock(gxPre, mr.scaleGrad(gyPre, pesc), gzReph);
            % LIN labels (same pattern as writeEpiDiffusionRS_6shot_v2_0p5mm_dpg_v2.m)
            seq.addBlock(mr.makeLabel('SET', 'LIN', 0));

            % Non-flyback 2-echo readout
            if isDummyTR && (iSlice ==1)
                seq.addBlock(gx);
                seq.addBlock(mr.scaleGrad(gx, -1));
            else
                seq.addBlock(gx, adc);
                seq.addBlock(mr.makeLabel('INC', 'LIN', 1));
                seq.addBlock(mr.scaleGrad(gx, -1));  % don't acquire 2nd echo during receive gain calibration
            end

            % Spoil and PE rephasing, and TR delay
            seq.addBlock(gxSpoil, mr.scaleGrad(gyPre, -pesc), gzSpoil);
            seq.addBlock(mr.makeDelay(delayTR));
        end
    end
end
%% Check sequence timing
[ok, error_report] = seq.checkTiming;
if (ok)
    fprintf('Timing check passed successfully\n');
else
    fprintf('Timing check failed! Error listing follows:\n');
    fprintf([error_report{:}]);
    fprintf('\n');
end

%% Output for execution and plot
seq.setDefinition('FOV', [fov fov sliceThickness*Nslices]);
seq.setDefinition('Name', 'gre2d');
seq.write([fn '.seq'])       % Write to pulseq file

seq.plot('timeRange', [21 30]*TR);
