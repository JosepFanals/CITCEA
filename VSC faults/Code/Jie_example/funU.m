function fun=funU(x,y_s,uth_s,zth,pavsc)

pdisp1=pavsc(1);
qdisp1=pavsc(2);
ivsc1max=pavsc(3);

zthx=real(zth);
zthy=imag(zth);

y1pp=y_s(1,1);
y1pn=y_s(1,2);
y1pz=y_s(1,3);
y1np=y_s(2,1);
y1nn=y_s(2,2);
y1nz=y_s(2,3);
y1zp=y_s(3,1);
y1zn=y_s(3,2);
y1zz=y_s(3,3);

y1ppx=real(y1pp);
y1ppy=imag(y1pp);
y1pnx=real(y1pn);
y1pny=imag(y1pn);
y1pzx=real(y1pz);
y1pzy=imag(y1pz);
y1npx=real(y1np);
y1npy=imag(y1np);
y1nnx=real(y1nn);
y1nny=imag(y1nn);
y1nzx=real(y1nz);
y1nzy=imag(y1nz);
y1zpx=real(y1zp);
y1zpy=imag(y1zp);
y1znx=real(y1zn);
y1zny=imag(y1zn);
y1zzx=real(y1zz);
y1zzy=imag(y1zz);

uthpx=real(uth_s(1));
uthpy=imag(uth_s(1));
uthnx=real(uth_s(2));
uthny=imag(uth_s(2));
uthzx=real(uth_s(3));
uthzy=imag(uth_s(3));


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
pcon1=x(25);
pcos1=x(26);
psin1=x(27);
qcon1=x(28);
qcos1=x(29);
qsin1=x(30);


fun(1)=pcon1-pdisp1; %USS
fun(2)=qcon1-qdisp1;
fun(3)=ivsc1nx;
fun(4)=ivsc1ny;
fun(5)=ivsc1zx;
fun(6)=ivsc1zy;

fun(7)=pcon1-(u1px*ivsc1px+u1py*ivsc1py+u1nx*ivsc1nx*u1ny*ivsc1ny);
fun(8)=pcos1-(u1px*ivsc1nx+u1py*ivsc1ny+u1nx*ivsc1px+u1ny*ivsc1py);
fun(9)=psin1-(u1px*ivsc1ny-u1py*ivsc1nx-u1nx*ivsc1py+u1ny*ivsc1px);
fun(10)=qcon1-(-u1px*ivsc1py+u1py*ivsc1px-u1nx*ivsc1ny+u1ny*ivsc1nx);
fun(11)=qcos1-(-u1px*ivsc1ny+u1py*ivsc1nx-u1nx*ivsc1py+u1ny*ivsc1px);
fun(12)=qsin1-(u1px*ivsc1nx+u1py*ivsc1ny-u1nx*ivsc1px-u1ny*ivsc1py);

fun(13)=i1px-(ivsc1px+ithpx);
fun(14)=i1py-(ivsc1py+ithpy);
fun(15)=i1nx-(ivsc1nx+ithnx);
fun(16)=i1ny-(ivsc1ny+ithny);
fun(17)=i1zx-(ivsc1zx+ithzx);
fun(18)=i1zy-(ivsc1zy+ithzy);

fun(19)=uthpx-(ithpx*zthx-ithpy*zthy+u1px);
fun(20)=uthpy-(ithpx*zthy+ithpy*zthx+u1py);
fun(21)=uthnx-(ithnx*zthx-ithny*zthy+u1nx);
fun(22)=uthny-(ithnx*zthy+ithny*zthx+u1ny);
fun(23)=uthzx-(ithzx*zthx-ithzy*zthy+u1zx);
fun(24)=uthzy-(ithzx*zthy+ithzy*zthx+u1zy);

%I=YU
fun(25)=i1px-(y1ppx*u1px-y1ppy*u1py)-(y1pnx*u1nx-y1pny*u1ny)-(y1pzx*u1zx-y1pzy*u1zy);
fun(26)=i1py-(y1ppx*u1py+y1ppy*u1px)-(y1pnx*u1ny+y1pny*u1nx)-(y1pzx*u1zy+y1pzy*u1zx);
fun(27)=i1nx-(y1npx*u1px-y1npy*u1py)-(y1nnx*u1nx-y1nny*u1ny)-(y1nzx*u1zx-y1nzy*u1zy);
fun(28)=i1ny-(y1npx*u1py+y1npy*u1px)-(y1nnx*u1ny+y1nny*u1nx)-(y1nzx*u1zy+y1nzy*u1zx);
fun(29)=i1zx-(y1zpx*u1px-y1zpy*u1py)-(y1znx*u1nx-y1zny*u1ny)-(y1zzx*u1zx-y1zzy*u1zy);
fun(30)=i1zy-(y1zpx*u1py+y1zpy*u1px)-(y1znx*u1ny+y1zny*u1nx)-(y1zzx*u1zy+y1zzy*u1zx);



