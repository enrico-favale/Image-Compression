clear all
close all 
clc
% La FFT è ampiamente utilizzata per la riduzione del rumore e il filtraggio dei segnali, poiché è semplice
% isolare e manipolare particolari bande di frequenza. 
% Introduciamo un filtro di soglia FFT per la riduzione del rumore di una immagine con rumore gaussiano
% aggiunto. In questo esempio, si osserva che il rumore è particolarmente
% pronunciato nelle alte frequenze e 
% pertanto azzeriamo qualsiasi coefficiente di Fourier al di fuori di un dato raggio contenente basse frequenze.
A = imread('dog.jpg'); % lettura immagine
B = rgb2gray(A); % trasformazione immagine in scala di grigi
figure
imshow(B) % Plot image
title('originale')
% introduciamo del rumore gaussiano
Bnoise = B + uint8(300*randn(size(B)));
% trasformata di Fourier
Bt= fft2(double(Bnoise));
% mostrare immagine con rumore
figure
imshow(Bnoise)
title('immagine con rumore')
% introdurre una griglia della dimensione della matrice B
[nx,ny] = size(B);
[X,Y] = meshgrid(-ny/2+1:ny/2,-nx/2+1:nx/2);
% questo comando mette le frequenze basse al centro della matrice
Btshift = fftshift(Bt);
% calcolare un raggio al di fuori del quale azzerare le frequenze
R2 = 100;
% mettere uguale a zero gli indici con frequenze maggiori di un certo
% raggio
ind = (X.^2 + Y.^2)>R2^2;
% azzerare le frequenze che corrispondono ad indice ind
Btshiftfilt = Btshift;           % Copia lo spettro originale
Btshiftfilt(ind) = 0;             % Azzera le frequenze selezionate
% riordiniamo le frequenze come erano prima
Btfilt = ifftshift(Btshiftfilt);
% ritorniamo nello spazio fisico
Bfilt = real(ifft2(Btfilt)); 
figure
imagesc(uint8(real(Bfilt))) % Filtered image
title('immagine filtrata')
colormap gray
set(gcf,'Position',[100 100 600 800])