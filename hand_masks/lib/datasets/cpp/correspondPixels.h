#pragma once
/*
  E = correspondPixels(E1, E2, h, w, maxDist, outlierCost, match1, match2)
  E1 -- input binary edge map 1
  E2 -- input binary edge map 2
  h -- height of the map
  w -- width of the map
  maxDist -- max distance for correspondence
  outlierCost -- penalty for outlier
  match1 -- matched result for input 1
  match2 -- matched result for input 2
*/
  
void c_correspondPixels(double *E1, double *E2, 
    const int h, const int w, const double maxDist, const double outlierCost, 
    double* match1, double* match2);