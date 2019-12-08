
[x1,fs]=audioread('p232_013_W.wav');
[x4,fs]=audioread('p232_103_W.wav');
[x7,fs]=audioread('p257_081_W.wav');
[x10,fs]=audioread('p257_101_W.wav');
[x13,fs]=audioread('p257_125_W.wav');

[x2,fs]=audioread('enhanced_p232_013__mixed_conditioned0_7dB.wav');
[x5,fs]=audioread('enhanced_p232_103__mixed_conditioned0_7dB.wav');
[x8,fs]=audioread('enhanced_p257_081__mixed_conditioned0_6dB.wav');
[x11,fs]=audioread('enhanced_p257_101__mixed_conditioned0_7dB.wav');
[x14,fs]=audioread('enhanced_p257_125__mixed_conditioned0_6dB.wav');

[x3,fs]=audioread('p232_013__cleaned_e145_snr_15.000000_p0.wav');
[x6,fs]=audioread('p232_103__cleaned_e145_snr_16.000000_p0.wav');
[x9,fs]=audioread('p257_081__cleaned_e145_snr_16.000000_p0.wav');
[x12,fs]=audioread('p257_101__cleaned_e145_snr_18.000000_p0.wav');
[x15,fs]=audioread('p257_125__cleaned_e145_snr_18.000000_p0.wav');

% subplot(321);plot(y);xlabel('时间');ylabel('幅度');title('原始信号的波形');
% subplot(323);plot(x);xlabel('时间');ylabel('幅度');title('观测信号的波形');
% subplot(325);plot(s);xlabel('时间');ylabel('幅度');title('增强信号的波形');
subplot(5,3,1);myspectrogram(x1,fs);colormap(jet);time=(0:length(x1)-1)/fs;axis([0 max(time*1000) 0 8000]);title('Spectrogram of Wiener-Signal');
subplot(5,3,2);myspectrogram(x2,fs);colormap(jet);time=(0:length(x2)-1)/fs;axis([0 max(time*1000) 0 8000]);title('Spectrogram of MSDGAN-Signal');
subplot(5,3,3);myspectrogram(x3,fs);colormap(jet);time=(0:length(x3)-1)/fs;axis([0 max(time*1000) 0 8000]);title('Spectrogram of Wavenet-Signal');
subplot(5,3,4);myspectrogram(x4,fs);colormap(jet);time=(0:length(x4)-1)/fs;axis([0 max(time*1000) 0 8000]);
subplot(5,3,5);myspectrogram(x5,fs);colormap(jet);time=(0:length(x5)-1)/fs;axis([0 max(time*1000) 0 8000]);
subplot(5,3,6);myspectrogram(x6,fs);colormap(jet);time=(0:length(x6)-1)/fs;axis([0 max(time*1000) 0 8000]);
subplot(5,3,7);myspectrogram(x7,fs);colormap(jet);time=(0:length(x7)-1)/fs;axis([0 max(time*1000) 0 8000]);
subplot(5,3,8);myspectrogram(x8,fs);colormap(jet);time=(0:length(x8)-1)/fs;axis([0 max(time*1000) 0 8000]);
subplot(5,3,9);myspectrogram(x9,fs);colormap(jet);time=(0:length(x9)-1)/fs;axis([0 max(time*1000) 0 8000]);
subplot(5,3,10);myspectrogram(x10,fs);colormap(jet);time=(0:length(x10)-1)/fs;axis([0 max(time*1000) 0 8000]);
subplot(5,3,11);myspectrogram(x11,fs);colormap(jet);time=(0:length(x11)-1)/fs;axis([0 max(time*1000) 0 8000]);
subplot(5,3,12);myspectrogram(x12,fs);colormap(jet);time=(0:length(x12)-1)/fs;axis([0 max(time*1000) 0 8000]);
subplot(5,3,13);myspectrogram(x13,fs);colormap(jet);time=(0:length(x13)-1)/fs;axis([0 max(time*1000) 0 8000]);xlabel('Time');
subplot(5,3,14);myspectrogram(x14,fs);colormap(jet);time=(0:length(x14)-1)/fs;axis([0 max(time*1000) 0 8000]);xlabel('Time')
subplot(5,3,15);myspectrogram(x15,fs);colormap(jet);time=(0:length(x15)-1)/fs;axis([0 max(time*1000) 0 8000]);xlabel('Time')

