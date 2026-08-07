%% RF excitation slice profile simulation (Bloch equations)
% Sinc 90° excitation: fixed duration, TBWP, slice thickness, and PSF layer count.
% Uses Bloch simulation with slice-select gradient during RF.
%
% Requires Pulseq MATLAB in path (e.g. addpath(genpath('Pulseq_Qiang_v151-main/matlab'))).

clear; close all; clc;

base_dir = fileparts(mfilename('fullpath'));
pulseq_path = fullfile(base_dir, 'Pulseq_Qiang_v151-main', 'matlab');
if exist(pulseq_path, 'dir')
    addpath(genpath(pulseq_path));
end

%% Protocol parameters
lims = mr.opts('MaxGrad', 70, 'GradUnit', 'mT/m', ...
    'MaxSlew', 150, 'SlewUnit', 'T/m/s', ...
    'rfRingdownTime', 10e-6, 'rfDeadtime', 100e-6, 'B0', 2.89);

thickness       = 6e-3;      % slice thickness (m)
tRFex           = 3e-3;      % RF excitation duration (s)
tbwp_exc        = 4;         % sinc time-bandwidth product
sinc_apodization = 0.5;      % Hanning-style apodization in mr.makeSincPulse
gamma           = 42.576e6;  % Hz/T

num_depth_layers = 12;
spa_res_mm = thickness * 1e3 / (num_depth_layers - 1);  % span nominal slice edge-to-edge

b1_plus_amplitude_scale = 1;

bw_exc_hz = tbwp_exc / tRFex;  % Hz; BW = TBWP / duration for sinc

% Spatial grid for profile (mm)
z_mm = linspace(-20, 20, 801);
z_m  = z_mm * 1e-3;

colors = [0 0.45 0.74; 0.85 0.33 0.10];
leg_exc = sprintf('Sinc 90° TBWP=%d', tbwp_exc);
ttl_b1_suffix = sprintf(' — sim: %.0f%% B_1^+ amplitude', 100 * b1_plus_amplitude_scale);

fprintf('\nExcitation pulse design\n');
fprintf('  Duration = %.3f ms, TBWP = %d, BW = %.1f Hz\n', tRFex*1e3, tbwp_exc, bw_exc_hz);
fprintf('  Slice thickness = %.1f mm, Gz = %.3f mT/m\n', thickness*1e3, (bw_exc_hz/thickness/gamma)*1e3);
fprintf('  PSF layers = %d, spacing = %.4f mm\n', num_depth_layers, spa_res_mm);

%% Sinc 90° excitation — Bloch + optional simRf cross-check
[rf, gz] = mr.makeSincPulse(pi/2, 'system', lims, 'Duration', tRFex, ...
    'SliceThickness', thickness, 'apodization', sinc_apodization, ...
    'timeBwProduct', tbwp_exc, 'use', 'excitation');
printRfPowerSummary(sprintf('Sinc 90  TBWP=%2d', tbwp_exc), rf, gamma);
printRfBandwidthSummary('Sinc 90  (FFT)', rf, bw_exc_hz);

[~, Mxy_exc] = simulateSliceSelectiveRf(rf, gz, z_m, [0; 0; 1], b1_plus_amplitude_scale);
Mxy_sinc = abs(Mxy_exc);
Mxy_sinc = Mxy_sinc / max(Mxy_sinc);

[~, Mxy_simrf, F_exc] = mr.simRf(rf);
Mxy_simrf_sinc = interp1(F_exc / gz.amplitude, abs(Mxy_simrf), z_m, 'linear', 0);
Mxy_simrf_sinc = Mxy_simrf_sinc / max(Mxy_simrf_sinc);

%% PSF weights from excitation profile (same recipe as Python loader)
center_offset = (num_depth_layers - 1) / 2;
z_layer_mm = spa_res_mm * ((0:num_depth_layers-1) - center_offset);
profile_at_layers = interp1(z_mm(:), Mxy_sinc(:), z_layer_mm(:), 'linear', 0);
profile_at_layers = max(profile_at_layers, 0);
weights_psf = profile_at_layers / sum(profile_at_layers);

fprintf('\nPSF weighting (excitation |M_{xy}|)\n');
fprintf('  Layer positions (mm): %s\n', mat2str(z_layer_mm, 6));
fprintf('  Weights: %s\n', mat2str(weights_psf(:)', 8));
fprintf('  Sum of weights: %.10f\n', sum(weights_psf));

rf_profile_mat = fullfile(base_dir, 'rf_slice_profile_exc_sinc_6mm_3ms.mat');
save(rf_profile_mat, 'z_mm', 'Mxy_sinc', 'Mxy_simrf_sinc', 'weights_psf', ...
    'z_layer_mm', 'spa_res_mm', 'num_depth_layers', 'thickness', 'tRFex', ...
    'tbwp_exc', 'bw_exc_hz', '-v7');
fprintf('  Saved: %s\n', rf_profile_mat);

%% Plot: excitation profile + PSF weights
half_exc_mm = 0.5 * thickness * 1e3;

figure('Color', 'w', 'Position', [160 160 560 760]);
hold on;
bh = barh(z_layer_mm, weights_psf(:), 'FaceColor', [0.65 0.78 0.95], ...
    'EdgeColor', [0.25 0.25 0.25], 'BarWidth', 0.72);
if isprop(bh, 'FaceAlpha')
    bh.FaceAlpha = 0.55;
end
hbox = plotIdealBoxVertical(z_mm, half_exc_mm);
hProf = plot(Mxy_sinc, z_mm, '-', 'LineWidth', 1.8, 'Color', colors(1,:));
addNominalSliceLinesVertical(thickness);
hold off;
ylim([-4 4]);
xlim([0 1.08]);
grid on;
xlabel('Normalized |M_{xy}|');
ylabel('Position (mm)');
set(gca, 'FontSize', 11);
title(sprintf('Excitation profile + PSF weights — %.1f mm, %.1f ms, TBWP=%d%s', ...
    thickness*1e3, tRFex*1e3, tbwp_exc, ttl_b1_suffix));
legend([bh, hbox, hProf], ...
    {sprintf('PSF weights (%d layers)', num_depth_layers), 'Ideal (rectangular)', leg_exc}, ...
    'Location', 'eastoutside');

%% Plot: full range and zoom — Bloch vs simRf
figure('Color', 'w', 'Position', [100 100 1100 420]);
tiledlayout(1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

nexttile;
hold on;
plotIdealBox(z_mm, half_exc_mm);
plot(z_mm, Mxy_sinc, '-', 'LineWidth', 1.8, 'Color', colors(1,:));
plot(z_mm, Mxy_simrf_sinc, '--', 'LineWidth', 1.5, 'Color', colors(2,:));
addNominalSliceLines(thickness);
hold off; grid on;
xlabel('Position (mm)'); ylabel('|M_{xy}| (normalized)');
legend({'Ideal (rectangular)', 'Bloch + G_z', 'mr.simRf'}, 'Location', 'best');
set(gca, 'FontSize', 11);
title(sprintf('90° excitation — full range (BW=%.0f Hz)%s', bw_exc_hz, ttl_b1_suffix));

nexttile;
hold on;
plotIdealBox(z_mm, half_exc_mm);
plot(z_mm, Mxy_sinc, '-', 'LineWidth', 1.8, 'Color', colors(1,:));
plot(z_mm, Mxy_simrf_sinc, '--', 'LineWidth', 1.5, 'Color', colors(2,:));
addNominalSliceLines(thickness);
hold off; xlim([-4 4]); grid on;
xlabel('Position (mm)'); ylabel('|M_{xy}| (normalized)');
legend({'Ideal (rectangular)', 'Bloch + G_z', 'mr.simRf'}, 'Location', 'best');
set(gca, 'FontSize', 11);
title(sprintf('90° excitation — enlarged (%.1f mm slice)%s', thickness*1e3, ttl_b1_suffix));

%% --- local helpers ---

function h = plotIdealBox(z_mm, half_width_mm)
    zmin = min(z_mm(:));
    zmax = max(z_mm(:));
    a = half_width_mm;
    hx = plot([zmin, -a, -a, a, a, zmax], [0, 0, 1, 1, 0, 0], 'r-', 'LineWidth', 2);
    if nargout > 0
        h = hx;
    end
end

function h = plotIdealBoxVertical(z_mm, half_width_mm)
    zmin = min(z_mm(:));
    zmax = max(z_mm(:));
    a = half_width_mm;
    hx = plot([0, 0, 1, 1, 0, 0], [zmin, -a, -a, a, a, zmax], 'r-', 'LineWidth', 2);
    if nargout > 0
        h = hx;
    end
end

function [Mz, Mxy] = simulateSliceSelectiveRf(rf, gz, z_m, M0, b1_amplitude_scale)
    if nargin < 5 || isempty(b1_amplitude_scale)
        b1_amplitude_scale = 1;
    end
    B1_Hz = rf.signal(:) .* exp(1i * rf.phaseOffset) * b1_amplitude_scale;
    t_rf  = rf.t(:);
    if length(t_rf) < 2
        dt = 1e-6;
    else
        dt = t_rf(2) - t_rf(1);
    end
    t_block = rf.delay + t_rf;
    Gz_t = sampleTrapezoidAt(gz, t_block);

    Mz = zeros(length(z_m), 1);
    Mxy = zeros(length(z_m), 1);
    for iz = 1:length(z_m)
        M = M0;
        for k = 1:length(t_rf)
            delta_f_Hz = Gz_t(k) * z_m(iz);
            omega_eff = 2*pi * [real(B1_Hz(k)); imag(B1_Hz(k)); delta_f_Hz];
            alpha = norm(omega_eff) * dt;
            if alpha > 1e-12
                M = blochRotate(M, omega_eff/norm(omega_eff), alpha);
            end
        end
        Mz(iz) = M(3);
        Mxy(iz) = M(1) + 1i*M(2);
    end
end

function printRfBandwidthSummary(label, rf, target_bw_hz)
    [bw_fft, fc] = mr.calcRfBandwidth(rf);
    fprintf('%-18s | FFT BW = %8.1f Hz (fc = %8.1f Hz)', label, bw_fft, fc);
    if nargin >= 3 && ~isempty(target_bw_hz)
        fprintf(' | target = %8.1f Hz | delta = %+.1f Hz', target_bw_hz, bw_fft - target_bw_hz);
    end
    fprintf('\n');
end

function printRfPowerSummary(label, rf, gamma)
    [total_energy, peak_pwr, rf_rms] = mr.calcRfPower(rf);
    peak_b1_uT = sqrt(peak_pwr) / gamma * 1e6;
    rf_rms_uT = rf_rms / gamma * 1e6;
    fprintf('%-18s | E = %10.3f | Peak = %10.3f | B1pk = %7.3f uT | RMS = %9.3f Hz (%7.3f uT)\n', ...
        label, total_energy, peak_pwr, peak_b1_uT, rf_rms, rf_rms_uT);
end

function addNominalSliceLines(thickness_m)
    x_edge_mm = 0.5 * thickness_m * 1e3;
    h1 = xline(-x_edge_mm, 'r:', 'LineWidth', 1.2);
    h2 = xline(+x_edge_mm, 'r:', 'LineWidth', 1.2);
    h1.Annotation.LegendInformation.IconDisplayStyle = 'off';
    h2.Annotation.LegendInformation.IconDisplayStyle = 'off';
end

function addNominalSliceLinesVertical(thickness_m)
    y_edge_mm = 0.5 * thickness_m * 1e3;
    h1 = yline(-y_edge_mm, 'r:', 'LineWidth', 1.2);
    h2 = yline(+y_edge_mm, 'r:', 'LineWidth', 1.2);
    h1.Annotation.LegendInformation.IconDisplayStyle = 'off';
    h2.Annotation.LegendInformation.IconDisplayStyle = 'off';
end

function g = sampleTrapezoidAt(gz, t)
    t = t(:);
    d = gz.delay;
    r = gz.riseTime;
    f = gz.flatTime;
    fl = gz.fallTime;
    a = gz.amplitude;

    g = zeros(size(t));
    for i = 1:length(t)
        ti = t(i) - d;
        if ti < 0 || ti >= r + f + fl
            g(i) = 0;
        elseif ti < r
            g(i) = a * (ti / r);
        elseif ti < r + f
            g(i) = a;
        else
            g(i) = a * (1 - (ti - r - f) / fl);
        end
    end
end

function M_out = blochRotate(M, n, alpha)
    n = n(:);
    cx = [0 -n(3) n(2); n(3) 0 -n(1); -n(2) n(1) 0];
    R = eye(3)*cos(alpha) + sin(alpha)*cx + (1 - cos(alpha))*(n*n');
    M_out = R * M(:);
end
