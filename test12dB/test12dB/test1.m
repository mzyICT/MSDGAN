[x1,fs]=audioread('p232_030__clean.wav');
[x4,fs]=audioread('p232_054__clean.wav');
[x7,fs]=audioread('p232_066__clean.wav');
[x10,fs]=audioread('p257_289__clean.wav');
[x13,fs]=audioread('p257_369__clean.wav');

[x2,fs]=audioread('p232_030__mixed_conditioned0_12dB.wav');
[x5,fs]=audioread('p232_054__mixed_conditioned0_12dB.wav');
[x8,fs]=audioread('p232_066__mixed_conditioned0_12dB.wav');
[x11,fs]=audioread('p257_289__mixed_conditioned0_12dB.wav');
[x14,fs]=audioread('p257_369__mixed_conditioned0_11dB.wav');

[x3,fs]=audioread('p232_030__cleaned_e145_snr_21.000000_p0.wav');
[x6,fs]=audioread('p232_054__cleaned_e145_snr_21.000000_p0.wav');
[x9,fs]=audioread('p232_066__cleaned_e145_snr_22.000000_p0.wav');
[x12,fs]=audioread('p257_289__cleaned_e145_snr_18.000000_p0.wav');
[x15,fs]=audioread('p257_369__cleaned_e145_snr_20.000000_p0.wav');

% subplot(321);plot(y);xlabel('时间');ylabel('幅度');title('原始信号的波形');
% subplot(323);plot(x);xlabel('时间');ylabel('幅度');title('观测信号的波形');
% subplot(325);plot(s);xlabel('时间');ylabel('幅度');title('增强信号的波形');
subplot(5,3,1);myspectrogram(x1,fs);colormap(jet);time=(0:length(x1)-1)/fs;axis([0 max(time*1000) 0 8000]);title('Spectrogram of Clean Signal');ylabel('ID:232-030-Frequency');
subplot(5,3,2);myspectrogram(x2,fs);colormap(jet);time=(0:length(x2)-1)/fs;axis([0 max(time*1000) 0 8000]);title('Spectrogram of Noisy Signal');
subplot(5,3,3);myspectrogram(x3,fs);colormap(jet);time=(0:length(x3)-1)/fs;axis([0 max(time*1000) 0 8000]);title('Spectrogram of Wavenet-Signal');
subplot(5,3,4);myspectrogram(x4,fs);colormap(jet);time=(0:length(x4)-1)/fs;axis([0 max(time*1000) 0 8000]);ylabel('ID:232-054-Frequency');
subplot(5,3,5);myspectrogram(x5,fs);colormap(jet);time=(0:length(x5)-1)/fs;axis([0 max(time*1000) 0 8000]);
subplot(5,3,6);myspectrogram(x6,fs);colormap(jet);time=(0:length(x6)-1)/fs;axis([0 max(time*1000) 0 8000]);
subplot(5,3,7);myspectrogram(x7,fs);colormap(jet);time=(0:length(x7)-1)/fs;axis([0 max(time*1000) 0 8000]);ylabel('ID:232-066-Frequency');
subplot(5,3,8);myspectrogram(x8,fs);colormap(jet);time=(0:length(x8)-1)/fs;axis([0 max(time*1000) 0 8000]);
subplot(5,3,9);myspectrogram(x9,fs);colormap(jet);time=(0:length(x9)-1)/fs;axis([0 max(time*1000) 0 8000]);
subplot(5,3,10);myspectrogram(x10,fs);colormap(jet);time=(0:length(x10)-1)/fs;axis([0 max(time*1000) 0 8000]);ylabel('ID:257-289-Frequency');
subplot(5,3,11);myspectrogram(x11,fs);colormap(jet);time=(0:length(x11)-1)/fs;axis([0 max(time*1000) 0 8000]);
subplot(5,3,12);myspectrogram(x12,fs);colormap(jet);time=(0:length(x12)-1)/fs;axis([0 max(time*1000) 0 8000]);
subplot(5,3,13);myspectrogram(x13,fs);colormap(jet);time=(0:length(x13)-1)/fs;axis([0 max(time*1000) 0 8000]);xlabel('Time');ylabel('ID:257-369-Frequency');
subplot(5,3,14);myspectrogram(x14,fs);colormap(jet);time=(0:length(x14)-1)/fs;axis([0 max(time*1000) 0 8000]);xlabel('Time')
subplot(5,3,15);myspectrogram(x15,fs);colormap(jet);time=(0:length(x15)-1)/fs;axis([0 max(time*1000) 0 8000]);xlabel('Time')



