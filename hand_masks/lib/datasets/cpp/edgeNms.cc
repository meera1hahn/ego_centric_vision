/*******************************************************************************
* From Structured Edge Detection Toolbox      Version 3.01
* Code written by Piotr Dollar, 2014.
* Licensed under the MSR-LA Full Rights License [see license.txt]
*******************************************************************************/
#include <math.h>
#include <stdlib.h>

// return I[x,y] via bilinear interpolation
inline float interp( float *I, const int h, const int w, float x, float y ) {
  x = x<0 ? 0 : (x>w-1.001 ? w-1.001 : x);
  y = y<0 ? 0 : (y>h-1.001 ? h-1.001 : y);
  int x0=(int)x, y0=(int)y, x1=x0+1, y1=y0+1;
  float dx0=x-x0, dy0=y-y0, dx1=1-dx0, dy1=1-dy0;
  return I[x0+y0*w]*dx1*dy1 + I[x1+y0*w]*dx0*dy1 +
    I[x0+y1*w]*dx1*dy0 + I[x1+y1*w]*dx0*dy0;
}

/*
  E = edgeNms(E, O, h, w, r, s, m)
  E -- original edge map
  O -- orientation map
  h -- height of the map
  w -- width of the map
  r -- radius for nms supr
  s -- radius for supr boundaries
  m -- multiplier for conservative supr
*/
void c_edgeNms(float *E, const float *O, const int h, const int w, 
            const int r, int s, const float m) {
  // removed multi-threading
  // in-place manipulation of E
  // row-majored array

  // copy E -> E0
  float *E0 = (float*)malloc(sizeof(float)*w*h);
  for( int y=0; y<h; y++ ) {
    for( int x=0; x<w; x++ ) {
      E0[x+y*w] = E[x+y*w];
    }   
  }

  // suppress edges where edge is stronger in orthogonal direction
  for( int y=0; y<h; y++ ) {
    for( int x=0; x<w; x++ ) {
      float e = E0[x+y*w]; if(!e) continue; e*=m;
      float coso=cos(O[x+y*w]); float sino=sin(O[x+y*w]);
      for( int d=-r; d<=r; d++ ) if( d ) {
        float e0 = interp(E0,h,w,x+d*coso,y+d*sino);
        if(e < e0) { E[x+y*w]=0; break; }
      }
    }   
  }

  // suppress noisy edge estimates near image boundaries
  s=s>w/2?w/2:s; s=s>h/2? h/2:s;
  for( int x=0; x<s; x++ ) for( int y=0; y<h; y++ ) {
    E[x+y*w]*=x/(float)s; E[(w-1-x)+y*w]*=x/(float)s; }
  for( int x=0; x<w; x++ ) for( int y=0; y<s; y++ ) {
    E[x+y*w]*=y/(float)s; E[x+(h-1-y)*w]*=y/(float)s; }
  free(E0);

}