#pragma once
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
  
void c_edgeNms(float *E, const float *O, const int h, const int w, const int r, int s, const float m);