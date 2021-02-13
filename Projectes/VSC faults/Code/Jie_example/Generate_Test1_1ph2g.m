clc;clear
close all

options=optimset;
options = optimset(options,'TolX',1e-10);
options = optimset(options,'TolFun',1e-10);
options = optimset(options,'MaxFunEvals',Inf);
options = optimset(options,'MaxIter',10000);
options = optimset(options,'Display','on');
options = optimset(options,'Algorithm','levenberg-marquardt');

%% Test System Parameters
w=50*2*pi; %frequency
uth=1;
uth_p=1; %positive
uth_n=0; %negative
uth_0=0; %zero
uth_s=[uth_p;uth_n;uth_0]; %sequence
zth=0.01+j*0.1;
rth=real(zth);
lth=imag(zth)/(100*pi);
rgrid=rth;
lgrid=lth;
zvsc=j*0.1; %not used in steady-state
lvsc=imag(zvsc)/(100*pi);%not used in steady-state
ivscmax=1;
imaxdq=ivscmax*sqrt(2);
ivscmax_ph=ivscmax/sqrt(3); %from single-phase pu to three-phase pu


%zsc_ph=[zsca;zscb;zscc];


pdisp=0.7; %VSC dispatched power
qdisp=0.5;

%Transformation

a=-1/2+j*sqrt(3)/2; % abc to sequential transformation
T=[1 1 1; a^2 a 1; a a^2 1];
T_n=1/3*[1 a a^2; 1 a^2 a; 1 1 1];

%Admittance matrix in abc
zsca=j*0.1;

yscaa=1/zsca;
yscbb=0;
ysccc=0;

%yscab=1/j*0.001;

y_abc=[yscaa,0,0;0,0,0;0,0,0]; %phase a to ground fault
y_s=T_n*y_abc*T;

z_th=[zth,0,0;0,zth,0;0,0,zth];
zth_s=T_n*z_th*T;

pavsc=[pdisp,qdisp,ivscmax];

%% solve
x0=[1,0,0,0,0,0 ...%voltage %initial value for solver
    1,0,0,0,0,0 ...%i1
    1,0,0,0,0,0 ...%ivsc1
    1,0,0,0,0,0 ...%ith
    pdisp,0,0,qdisp,0,0 ...%pqvsc
    ];

[x,fval,exitflag]=fsolve(@(x)funU(x,y_s,uth_s,zth,pavsc), x0,options);

u1px=x(1);
u1py=x(2);
u1nx=x(3);
u1ny=x(4);
u1zx=x(5);
u1zy=x(6);
i1px=x(7);
i1py=x(8);
i1nx=x(9);
i1ny=x(10);
i1zx=x(11);
i1zy=x(12);
ivsc1px=x(13);
ivsc1py=x(14);
ivsc1nx=x(15);
ivsc1ny=x(16);
ivsc1zx=x(17);
ivsc1zy=x(18);
ithpx=x(19);
ithpy=x(20);
ithnx=x(21);
ithny=x(22);
ithzx=x(23);
ithzy=x(24);


u1s=[u1px+j*u1py;u1nx+j*u1ny;u1zx+j*u1zy];
u1abc=1/sqrt(3)*T*u1s;
ivsc1=ivsc1px+j*ivsc1py;

ifts=[i1px+j*i1py;i1nx+j*i1ny;i1zx+j*i1zy];
iftabc=1/sqrt(3)*T*ifts;

u1a_mag=abs(u1abc(1))
u1a_ang=atan(imag(u1abc(1))/real(u1abc(1)))/pi*180
u1b_mag=abs(u1abc(2))
u1b_ang=atan(imag(u1abc(2))/real(u1abc(2)))/pi*180
u1c_mag=abs(u1abc(3))
u1c_ang=atan(imag(u1abc(3))/real(u1abc(3)))/pi*180

ifta_mag=abs(iftabc(1))
ifta_ang=atan(imag(iftabc(1))/real(iftabc(1)))/pi*180

ivsc1mag=abs(ivsc1)
ivsc1ang=atan(ivsc1py/ivsc1px)/pi*180

pcon1=x(25)
pcos1=x(26)
psin1=x(27)
qcon1=x(28)
qcos1=x(29)
qsin1=x(30)