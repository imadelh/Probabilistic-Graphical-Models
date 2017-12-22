clear
close all
clc

% This will load the matrix V
load('./attfaces.mat');

height = 112;
width = 92;

numVisualize = 400; %must be a square
V_visual = zeros(height*sqrt(numVisualize),width*sqrt(numVisualize));

figure;
ix = 1;
ix1 = 1;
for i = 1:sqrt(numVisualize)
    ix2 = 1;
    for j = 1:sqrt(numVisualize)
        curV = V(:,ix);
        curV = reshape(uint8(curV),[height, width]);
        V_visual(ix1:(ix1+height-1), ix2:(ix2+width-1)) = curV;
        ix2 = ix2 + width;
        ix = ix+1;
    end
    ix1 = ix1 + height;
end

imagesc(V_visual); axis ij
caxis([0 255]);
colormap(gray);

