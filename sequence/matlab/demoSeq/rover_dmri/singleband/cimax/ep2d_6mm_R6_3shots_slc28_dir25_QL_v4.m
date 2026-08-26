% 3-shot, R=6 per shot, dpg
% no blip b0 + AP b0 + PA b0 + diffusion scan

% Qiang Liu qliu30@mgh.harvard.edu
clc;
close all;
clear;

seq_file        =   'ep2d_6mm_TE74_R2_3shots_slc28_dir30_rot12.seq';
save_seq_file   =   1;
seq_pns         =   1;
seq_plot        =   0;
seq_report      =   0;
check_freq      =   0;
test_check      =   0;

%% Set system limits

B0field     =   2.89;
lims        =   mr.opts('MaxGrad',80,'GradUnit','mT/m',...
    'MaxSlew',130,'SlewUnit','T/m/s',...
    'rfRingdownTime',10e-6,'rfDeadtime',100e-6,'B0',B0field); % max slew rate: 200
lims1       =   mr.opts('MaxGrad',68,'GradUnit','mT/m',...  % MaxGrad 68
    'MaxSlew',80,'SlewUnit','T/m/s',...
    'rfRingdownTime',10e-6,'rfDeadtime',100e-6,'B0',B0field); % this lims1 is set for gz fat
lims2       =   mr.opts('MaxGrad',172,'GradUnit','mT/m',...  % MaxGrad 68
    'MaxSlew',90,'SlewUnit','T/m/s',...
    'rfRingdownTime',10e-6,'rfDeadtime',100e-6,'B0',B0field); % for diffusion gradients

%% Imaging parameter
table=load('/rfanfs/pnl-zorro/home/ql087/lq/Pulseq_V151_ROVER_dMRI/sequence/diffusion_table/30_dir_table.txt');
table=[table([1 1 1],:);  table];
bFactor=1000.*ones(1,size(table,1)); % s/mm^2
bFactor_scale=ones(1,length(bFactor));
bFactor_scale(:,1:3)=sqrt(10./bFactor(:,1:3)); 
diffusion_count     =   size(table,1);
% diffusion_count = 9; 

fov                 =   200e-3;
Nx                  =   400;
Nx_org              =   400;            % for gx pre-phasing calculation
Ny                  =   Nx;             % Define FOV and resolution (Nx)
thickness           =   6e-3;           % slice thinckness
Nslices             =   28;

recon               =   false;          % plot the traj for check
TE                  =   74e-3; % 50 for 1D PF 43 for 2D PF 5/8
TR                  =   4200e-3;

RSegment            =   3;              % multishot factor
R                   =   2.0;              % In-plane accerelation factor

Echotimeshift       =   1;              % for multishot
blip_sw            =   1; % dpg
ro_os               =   1.00;
readoutTime         =   13.6e-4; % 10.8 for 6/8 9.8e-4 for 5/8  13.4 for full
partFourierFactor           =  6/8;     % partial Fourier factor: 1: full sampling 0: start with ky=0
partFourierFactor_fe        =  1;     % frequency-encoding partial Fourier factor
Nx                  =   Nx * partFourierFactor_fe;

tRFex               =   3e-3;           % sec
tRFref              =   3e-3;           % sec
sat_ppm             =   -3.45;

deltak              =   1/fov;
deltaky             =   RSegment*R*deltak;
kWidth              =   Nx*deltak;
kWidth_org          =   Nx_org*deltak;

rotation_angles=[0:11]*pi/12;

%% Define Seq
seq             =   mr.Sequence(lims);      % Create a new sequence object
sat_freq            =   sat_ppm*1e-6*lims.B0*lims.gamma;
rf_fs               =   mr.makeGaussPulse(110*pi/180,'system',lims1,'Duration',8e-3,...
    'bandwidth',abs(sat_freq),'freqOffset',sat_freq,'use','saturation');
rf_fs.phaseOffset   =   -2*pi*rf_fs.freqOffset*mr.calcRfCenter(rf_fs); % compensate for the frequency-offset induced phase
gz_fs               =   mr.makeTrapezoid('z',lims1,'delay',mr.calcDuration(rf_fs),'Area',1/1e-4); % spoil up to 0.1mm

spoiler_amp=3*8*42.58*10e2;
est_rise=500e-6; % ramp time 280 us
est_flat=2500e-6; %duration 600 us

gp_r=mr.makeTrapezoid('x','amplitude',spoiler_amp,'riseTime',est_rise,'flatTime',est_flat,'system',lims1);
gp_p=mr.makeTrapezoid('y','amplitude',spoiler_amp,'riseTime',est_rise,'flatTime',est_flat,'system',lims1);
gp_s=mr.makeTrapezoid('z','amplitude',spoiler_amp,'riseTime',est_rise,'flatTime',est_flat,'system',lims1);

gn_r=mr.makeTrapezoid('x','amplitude',-spoiler_amp,'delay',mr.calcDuration(rf_fs), 'riseTime',est_rise,'flatTime',est_flat,'system',lims1);
gn_p=mr.makeTrapezoid('y','amplitude',-spoiler_amp,'delay',mr.calcDuration(rf_fs), 'riseTime',est_rise,'flatTime',est_flat,'system',lims1);
gn_s=mr.makeTrapezoid('z','amplitude',-spoiler_amp,'delay',mr.calcDuration(rf_fs), 'riseTime',est_rise,'flatTime',est_flat,'system',lims1);

[rf, gz, gzReph]    =   mr.makeSincPulse_QL_0(pi/2,'system',lims,'Duration',tRFex,...
    'SliceThickness',thickness,'apodization',0.5,'timeBwProduct',4,'use','excitation');
[rf180, gz180]      =   mr.makeSincPulse(pi,'system',lims,'Duration',tRFref,...
    'SliceThickness',thickness,'apodization',0.5,'timeBwProduct',4,'PhaseOffset',pi/2,'use','refocusing');
blip_dur            =   ceil(2*sqrt(deltaky/lims.maxSlew)/10e-6/2)*10e-6*2;
gy                  =   mr.makeTrapezoid('y',lims,'Area',-deltaky,'Duration',blip_dur);
extra_area          =   blip_dur/2*blip_dur/2*lims.maxSlew; % check unit!;
gx                  =   mr.makeTrapezoid('x',lims,'Area',kWidth+extra_area,'duration',readoutTime+blip_dur);
actual_area         =   gx.area-gx.amplitude/gx.riseTime*blip_dur/2*blip_dur/2/2-gx.amplitude/gx.fallTime*blip_dur/2*blip_dur/2/2;
gx.amplitude        =   gx.amplitude/actual_area*kWidth;
gx.area             =   gx.amplitude*(gx.flatTime + gx.riseTime/2 + gx.fallTime/2);
gx.flatArea         =   gx.amplitude*gx.flatTime;

extra_area_org      =   blip_dur/2*blip_dur/2*lims.maxSlew;
gx_org              =   mr.makeTrapezoid('x',lims,'Area',kWidth_org+extra_area_org);
actual_area_org     =   gx_org.area-gx_org.amplitude/gx_org.riseTime*blip_dur/2*blip_dur/2/2-gx_org.amplitude/gx_org.fallTime*blip_dur/2*blip_dur/2/2;
gx_org.amplitude    =   gx_org.amplitude/actual_area_org*kWidth_org;
gx_org.area         =   gx_org.amplitude*(gx_org.flatTime + gx_org.riseTime/2 + gx_org.fallTime/2);
gx_org.flatArea     =   gx_org.amplitude*gx_org.flatTime;

adcDwellNyquist     =   deltak/gx.amplitude/ro_os;

adcDwell            =   floor(adcDwellNyquist*1e7)*1e-7;
adcSamples          =   floor(readoutTime/adcDwell/4)*4; % on Siemens the number of ADC samples need to be divisible by 4
adc                 =   mr.makeAdc(adcSamples,'Dwell',adcDwell,'Delay',blip_dur/2);

time_to_center      =   adc.dwell*((adcSamples-1)/2+0.5); % Siemens samples in the center of the dwell period
adc.delay           =   round((gx.riseTime+gx.flatTime/2-time_to_center)*1e6)*1e-6; % adjust the delay to align the trajectory with the gradient. We have to aligh the delay to 1us

gy_parts                    =   mr.splitGradientAt(gy, blip_dur/2, lims);
[gy_blipup, gy_blipdown,~]  =   mr.align('right',gy_parts(1),'left',gy_parts(2),gx);
gy_blipdownup               =   mr.addGradients({gy_blipdown, gy_blipup}, lims);
% phase encoding and partial Fourier
Ny_pre      =   round((partFourierFactor-1/2)*Ny-1);  % PE steps prior to ky=0, excluding the central line
Ny_pre      =   round(Ny_pre/RSegment/R);
Ny_post     =   round(Ny/2+1); % PE lines after the k-space center including the central line
Ny_post     =   round(Ny_post/RSegment/R);
Ny_meas     =   Ny_pre+Ny_post;

gxPre           = mr.makeTrapezoid('x',lims1,'Area',-(gx.area-gx_org.area/2)); % QL
gyPre = mr.makeTrapezoid('y',lims1,'Area',(Ny_pre*deltaky));
[gxPre,gyPre]=mr.align('right',gxPre,'left',gyPre);
gyPre = mr.makeTrapezoid('y',lims1,'Area',gyPre.area,'Duration',mr.calcDuration(gxPre,gyPre));

if (mod(Ny_meas,2)==1)
    gxPre_post = mr.makeTrapezoid('x',lims1,'Area',-gx_org.area/2);
else
    gxPre_post = gxPre;
end

gyPre_post = mr.makeTrapezoid('y',lims1,'Area',(Ny_post-1)*deltaky);
[gxPre_post,gyPre_post]=mr.align('right',gxPre_post,'left',gyPre_post);
gyPre_post = mr.makeTrapezoid('y',lims1,'Area',gyPre_post.area,'Duration',mr.calcDuration(gxPre_post,gyPre_post));

if (mod(Ny_meas,2)==0)
    gxPre_post.amplitude=-gxPre_post.amplitude;
end

gxPre.delay=0;
gyPre.delay=0;
dur2end=(Ny_post-0.5)*mr.calcDuration(gx)+mr.calcDuration(gxPre_post,gyPre_post);

durationToCenter        = (Ny_pre + 0.5) * mr.calcDuration(gx);
rfCenterInclDelay       = rf.delay + mr.calcRfCenter(rf);
rf180centerInclDelay    = rf180.delay + mr.calcRfCenter(rf180);

delayTE1                = ceil((TE/2 - mr.calcDuration(rf,gz) - mr.calcDuration(gzReph) + rfCenterInclDelay - rf180centerInclDelay)/lims.gradRasterTime)*lims.gradRasterTime;
assert(delayTE1>=0);

delayTE2                = ceil((TE/2 - mr.calcDuration(rf180,gz180) + rf180centerInclDelay - durationToCenter)/lims.gradRasterTime)*lims.gradRasterTime;
delayTE2                = delayTE2 - mr.calcDuration(gxPre,gyPre);
assert(delayTE2>=0);

delayTR     =   ceil((TR/Nslices - mr.calcDuration(gp_r) - mr.calcDuration(gn_r) - TE - rfCenterInclDelay-dur2end)/seq.gradRasterTime)*seq.gradRasterTime; % v5
delayTR     =   round(delayTR,3);
dTR         =   mr.makeDelay(delayTR);

for islice_1=1:Nslices
    freqOffset_factor(islice_1)=islice_1-1-(Nslices-1)/2;
end
slic_indexS=[2:2:Nslices 1:2:Nslices];
interleaved_freqOffset_factor=freqOffset_factor(slic_indexS);

for iview=[1:12] % I typically split them to 1:3, 4:6, 7:9, and 10:12 if I am running 30 directions

    alpha_rad=rotation_angles(iview);
    rot_mat = [cos(alpha_rad), 0, sin(alpha_rad);0, 1, 0;-sin(alpha_rad), 0, cos(alpha_rad)];
    dpg=1;

    for idpg=1:2

        if (idpg==1)
            dpg = 1;
            real_ds_count = diffusion_count;
        else
            dpg = -1;
            real_ds_count = 2;
        end

        for rep = 1:real_ds_count
            if rep == 1
                pe_enable=eps;
            else
                pe_enable=1;
            end

            if rep ==3
                blip_sw=-1;
            else
                blip_sw=1;
            end

            Nmulti_end = RSegment;

            for Nmulti = 1:Nmulti_end

                if idpg==2 && Nmulti~=1   % only shot 1 will have dpg reference
                    continue;
                end

                small_delta=delayTE2-ceil(lims2.maxGrad/lims2.maxSlew/lims.gradRasterTime)*lims.gradRasterTime;
                big_delta=delayTE1+mr.calcDuration(rf180,gz180);
                g=sqrt(bFactor(rep)*1e6/bFactCalc(1,small_delta,big_delta));
                gr=ceil(g/lims2.maxSlew/lims.gradRasterTime)*lims.gradRasterTime;
                gDiff=mr.makeTrapezoid('z','amplitude',g,'riseTime',gr,'flatTime',small_delta-gr,'system',lims2); % v2

                %% split the diffusion gradient to 3 axis, new version based on Jon and Maxim's suggestions (QL) % v2

                g_x=g.*table(rep,1);
                g_y=g.*table(rep,2);
                g_z=g.*table(rep,3);

                if (table(rep,1)==0&&table(rep,2)==0&&table(rep,3)==0) % when the b-value=0, the section using Sepherical coorodinate system is wrong.
                    gDiff_x=gDiff; gDiff_x.channel='x';
                    gDiff_y=gDiff; gDiff_y.channel='y';
                    gDiff_z=gDiff; gDiff_z.channel='z';
                else
                    [azimuth,elevation,r] = cart2sph(g_x,g_y,g_z);
                    polar= -(pi/2-elevation);

                    Gr=mr.rotate('z',azimuth,mr.rotate('y',polar,gDiff));
                    if size(Gr,2)==3
                        gDiff_x=Gr{1,2};
                        gDiff_y=Gr{1,3};
                        gDiff_z=Gr{1,1};
                    else
                        if size(Gr,2)==2
                            diffusion_blank=find( table(rep,:)==0);
                            switch diffusion_blank
                                case 2
                                    gDiff_x=Gr{1,2};
                                    gDiff_z=Gr{1,1};
                                    gDiff_y=gDiff; gDiff_y.channel='y'; gDiff_y.amplitude=0; gDiff_y.area=0; gDiff_y.flatArea=0;
                                case 1
                                    gDiff_z=Gr{1,1};
                                    gDiff_y=Gr{1,2};
                                    gDiff_x=gDiff; gDiff_x.amplitude=0; gDiff_x.area=0; gDiff_x.flatArea=0;gDiff_x.channel='x';
                                case 3
                                    gDiff_x=Gr{1,2};
                                    gDiff_y=Gr{1,1};
                                    gDiff_z=gDiff; gDiff_z.amplitude=0; gDiff_z.area=0; gDiff_z.flatArea=0;gDiff_z.channel='z';
                            end
                        end
                    end
                end

                duration_diff=mr.calcDuration(gDiff);

                assert(duration_diff<=delayTE1);
                assert(duration_diff<=delayTE2);

                %% Calculate the echo time shift for multishot EPI (QL)
                actual_esp=gx.riseTime+gx.flatTime+gx.fallTime;
                TEShift=actual_esp/RSegment;
                TEShift=round(TEShift,5);
                TEShift_before_echo=(Nmulti-1)*TEShift;
                TEShift_before_echo=round(TEShift_before_echo/lims.blockDurationRaster)*lims.blockDurationRaster;

                if TEShift_before_echo ==0
                    TEShift_before_echo=eps; % apply the minimum duration for the no delay case
                end

                TEShift_after_echo=(RSegment-(Nmulti-1))*TEShift;
                TEShift_after_echo=round(TEShift_after_echo/lims.blockDurationRaster)*lims.blockDurationRaster;
                dETS_before=mr.makeDelay(TEShift_before_echo);
                dETS_after=mr.makeDelay(TEShift_after_echo);

                scale_gyPre_segment = (Ny_pre*deltaky - (Nmulti-1)*deltak*R ) / (Ny_pre*deltaky); % deltaky
                scale_gyPost_segment = ((Ny_post-1)*deltaky + (Nmulti-1)*deltak*R) / ((Ny_post-1)*deltaky);
                
                rot = mr.makeRotation(rot_mat);  % rotation event

                if test_check == 1
                    slc_end = 1;
                else
                    slc_end = Nslices;
                end
                %% Define sequence blocks
                for slc = 1:slc_end
                    if (~seq_plot)
                        seq.addBlock(gp_r,gp_p,gp_s);
                        seq.addBlock(rf_fs,gn_r,gn_p,gn_s);
                    end

                    rf.freqOffset           =   gz.amplitude*thickness*interleaved_freqOffset_factor(slc);
                    rf.phaseOffset=-2*pi*rf.freqOffset*mr.calcRfCenter(rf); % compensate for the slice-offset induced phase
                    rf180.freqOffset        =   gz180.amplitude*thickness*interleaved_freqOffset_factor(slc);
                    rf180.phaseOffset=pi/2-2*pi*rf180.freqOffset*mr.calcRfCenter(rf180); % compensate for the slice-offset induced phase

                    seq.addBlock(rf,gz,rot);
                    seq.addBlock(gzReph,rot);
                    seq.addBlock(mr.makeDelay(delayTE1),mr.scaleGrad(gDiff_x, bFactor_scale(1,rep)), mr.scaleGrad(gDiff_y, bFactor_scale(1,rep)), mr.scaleGrad(gDiff_z, bFactor_scale(1,rep)));
                    seq.addBlock(gz180, rf180,rot);
                    seq.addBlock(mr.makeDelay(delayTE2),mr.scaleGrad(gDiff_x, bFactor_scale(1,rep)), mr.scaleGrad(gDiff_y, bFactor_scale(1,rep)), mr.scaleGrad(gDiff_z, bFactor_scale(1,rep)));

                    if (Echotimeshift)
                        seq.addBlock(dETS_before); % echotimeshift for multishot
                    end

                    seq.addBlock(mr.scaleGrad(gxPre,dpg),mr.scaleGrad(gyPre,blip_sw*pe_enable*scale_gyPre_segment), rot);
                    for i=1:Ny_meas
                        if i==1
                            seq.addBlock(mr.scaleGrad(gx,dpg),mr.scaleGrad(gy_blipup, blip_sw*pe_enable),adc,rot); % Read the first line of k-space with a single half-blip at the end
                        elseif i==Ny_meas
                            seq.addBlock(mr.scaleGrad(gx,dpg),mr.scaleGrad(gy_blipdown, blip_sw*pe_enable),adc,rot); % Read the last line of k-space with a single half-blip at the beginning
                        else
                            seq.addBlock(mr.scaleGrad(gx,dpg),mr.scaleGrad(gy_blipdownup, blip_sw*pe_enable),adc,rot); % Read an intermediate line of k-space with a half-blip at the beginning and a half-blip at the end
                        end
                        if (mod(Ny_meas,2)==1)
                            if (i<Ny_meas)
                                gx.amplitude = -gx.amplitude;
                            end
                        else
                            gx.amplitude = -gx.amplitude;
                        end
                    end
                    seq.addBlock(mr.scaleGrad(gxPre_post,dpg),mr.scaleGrad(gyPre_post,blip_sw*pe_enable*scale_gyPost_segment), rot); % go back to k-space center

                    if (Echotimeshift)
                        seq.addBlock(dETS_after); % echotimeshift for multishot
                    end
                    seq.addBlock(dTR); % seperate
                end %slice loop
            end % for multishot
        end
    end
    disp(['rot view: ', num2str(iview)])
end % rotation view

%% check whether the timing of the sequence is correct

[ok, error_report]=seq.checkTiming;

if (ok)
    fprintf('Timing check passed successfully\n');
else
    fprintf('Timing check failed! Error listing follows:\n');
    fprintf([error_report{:}]);
    fprintf('\n');
end

%% do some visualizations

if seq_plot == 1
    tic
    fprintf('Plotting figures... ');
    seq.plot(); % plot sequence waveforms
    [ktraj_adc, t_adc, ktraj, t_ktraj, t_excitation, t_refocusing, slicepos, t_slicepos] = seq.calculateKspacePP();
    figure; plot(t_ktraj, ktraj'); % plot the entire k-space trajectory
    hold on; plot(t_adc,ktraj_adc(1,:),'.'); % and sampling points on the kx-axis

    figure; plot(ktraj(1,:),ktraj(2,:),'b'); % a 2D plot
    axis('equal'); % enforce aspect ratio for the correct trajectory display
    hold on; plot(ktraj_adc(1,:),ktraj_adc(2,:),'r.'); % plot the sampling points
    toc
end


%% prepare the sequence output for the scanner

seq.setDefinition('FOV', [fov fov thickness*Nslices]);
seq.setDefinition('Name', 'epi-diff');

if seq_pns == 1
    tic
    fprintf('Checking PNS... ');
    % pns < 80\% (human)
    [pns_ok, pns_n, pns_c, tpns] = seq.calcPNS('/rfanfs/pnl-zorro/home/ql087/lq/Tests/scanner_asc/MP_GradSys_P034_X60.asc'); % PRISMA-XR
    pns = max(pns_n(:));
    clear ktraj_adc t_adc ktraj t_ktraj t_excitation t_refocusing slicepos t_slicepos pns_n pns_c tpns
    toc
end

if save_seq_file == 1
    tic
    fprintf('Saving .seq file... ');
    seq.write(seq_file);
    save(strcat(seq_file(1:end-4),'_param'));
    toc
end

if check_freq == 1
    tic
    fprintf('Checking frequencies... ');
    sys = lims;
    gradSpectrum;
    toc
end

function b=bFactCalc(g, delta, DELTA)
sigma=1;
kappa_minus_lambda=1/3-1/2;
b= (2*pi * g * delta * sigma)^2 * (DELTA + 2*kappa_minus_lambda*delta);
end