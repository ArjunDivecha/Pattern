/* Exact valid-input chart renderer. No allocation; writes N uint8 images. */
#include <math.h>
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
static int pixel(double x,double lo,double hi,int rows){double f=(hi-x)/(hi-lo); if(f<0)f=0;if(f>1)f=1;return (int)nearbyint(f*(rows-1));}
static void line(uint8_t *img,int w,int r0,int c0,int r1,int c1){
 int dr=abs(r1-r0),dc=abs(c1-c0),sr=r1>=r0?1:-1,sc=c1>=c0?1:-1,err=dr-dc;
 for(;;){img[r0*w+c0]=255;if(r0==r1&&c0==c1)break;int e=2*err;if(e>-dc){err-=dc;r0+=sr;}if(e<dr){err+=dr;c0+=sc;}}
}
int render(const double *ohlcv,const double *tri,const int64_t *ends,int n,int window,int ma,uint8_t *output){
 const int height=64,rows=51,volrows=12,width=3*window;
 for(int j=0;j<n;j++){
  int64_t end=ends[j],start=end-window+1;uint8_t *img=output+(size_t)j*height*width;
  double base=tri[start],lo=INFINITY,hi=-INFINITY,vmax=0;
  if(!(base>0&&isfinite(base)))return j+1;
  for(int t=0;t<window;t++){
   int64_t k=start+t;double p=tri[k]/base,s=p/ohlcv[5*k+3];
   for(int a=0;a<4;a++){double v=a==3?p:ohlcv[5*k+a]*s;if(!isfinite(v))return j+1;if(v<lo)lo=v;if(v>hi)hi=v;}
   if(ohlcv[5*k+4]>vmax)vmax=ohlcv[5*k+4];
  }
  if(hi==lo)hi=lo+1e-8;if(vmax<=0)vmax=1;
  int prev=-1;
  for(int t=0;t<window;t++){
   int64_t k=start+t;double p=tri[k]/base,s=p/ohlcv[5*k+3];int c=3*t;
   int h=pixel(ohlcv[5*k+1]*s,lo,hi,rows),l=pixel(ohlcv[5*k+2]*s,lo,hi,rows);
   for(int r=h;r<=l;r++)img[r*width+c+1]=255;
   img[pixel(ohlcv[5*k]*s,lo,hi,rows)*width+c]=255;img[pixel(p,lo,hi,rows)*width+c+2]=255;
   double sum=0;for(int m=0;m<ma;m++)sum+=tri[k-ma+1+m]/base;
   int r=pixel(sum/ma,lo,hi,rows);img[r*width+c+1]=255;if(prev>=0)line(img,width,prev,c-2,r,c+1);prev=r;
   int vh=(int)nearbyint(ohlcv[5*k+4]/vmax*volrows);for(int vr=height-vh;vr<height;vr++)img[vr*width+c+1]=255;
  }
 }
 return 0;
}
