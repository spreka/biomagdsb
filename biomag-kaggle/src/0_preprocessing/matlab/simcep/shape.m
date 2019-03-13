%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SHAPE Function for random shape model (for details see manuscript)
% Input:  (1) shape parameter (alpha)
%         (2) shape parameter (beta)
%         (3) radius of object
% Output: (1) shape as a binary image
%
% (C) 22.2.2007 Antti Lehmussola
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function[out] = shape(s1,s2,r)

% step = 0.2;
% t = (0:step:1)'*2*pi;
t = linspace(0,1,8)'*2*pi;

% r1 = rand(size(t))-0.5;
% r2 = rand(size(t))-0.5;

% rs = (sin(circshift(t,randi(length(t))))/2+1.5);
% rs = 1;

rs = sin( randi(4)/2 .* circshift(t,randi(length(t))) )/ 2+ 1.5 + rand() ;


t1 = s1.*(2*rand(size(t))-1)+(sin(t+s2.*(2*rand(size(t))-1))).*rs;
t2 = s1.*(2*rand(size(t))-1)+(cos(t+s2.*(2*rand(size(t))-1))).*rs;
t1(end) = t1(1);
t2(end) = t2(1);

t1 = (t1-mean(t1))/(max(t1)-min(t1))*r*2;
t2 = (t2-mean(t2))/(max(t2)-min(t2))*r*2;

% max(t1)
% max(t2)

object = [t2';t1'];

pp_nuc = cscvn(object);
object = ppval(pp_nuc, linspace(0,max(pp_nuc.breaks),1000));
	
% object = [r*object(1,:); r*(rand()+1)*object(2,:)];
% object = object*r;



object(1,:) = object(1,:) - min(object(1,:));
object(2,:) = object(2,:) - min(object(2,:));
object = round(object)+1;

% object(1,:) = object(1,:) - mean(object(1,:));
% object(2,:) = object(2,:) - mean(object(2,:));

% angle = randi(360)*2*pi/180;
% object(1,:) = object(1,:) .* (sin(angle)) ;
% object(2,:) = object(2,:) .* (cos(angle));

% object(1,:) = object(1,:) - min(object(1,:));
% object(2,:) = object(2,:) - min(object(2,:));
% object = round(object)+1;

I = zeros(max(round(object(1,:))),max(round(object(2,:))));
BW = roipoly(I,object(2,:),object(1,:));



% BW2 = imrotate(BW,randi(360),'bicubic');

% figure(1); imagesc(BW2); axis image;

emptyRows = ~any(BW,2);
BW(emptyRows,:) = [];

emptyCols = ~any(BW,1);
BW(:,emptyCols) = [];

out = BW;

% figure(1); imagesc(BW); axis image;
