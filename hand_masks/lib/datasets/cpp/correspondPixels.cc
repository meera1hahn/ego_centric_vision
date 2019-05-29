#include <string.h>
#include "Matrix.hh"
#include "csa.hh"
#include "match.hh"
#include "correspondPixels.h"


void c_correspondPixels(double *E1, double *E2, 
    const int h, const int w, const double maxDist, const double outlierCost, 
    double* match1, double* match2) {

    const double idiag = sqrt( h*h + w*w );
    const double oc = outlierCost*maxDist*idiag;
    Matrix m1, m2;
    double cost = matchEdgeMaps(
        Matrix(w,h,E1), Matrix(w,h,E2),
        maxDist*idiag, oc,
        m1, m2);
    
    // set output arguments
    memcpy(match1,m1.data(),m1.numel()*sizeof(double));
    memcpy(match2,m2.data(),m2.numel()*sizeof(double));

}
