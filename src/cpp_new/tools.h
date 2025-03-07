/*
  -------------------------------------------------------------------
  
  Copyright (C) 2015-2020, Andrew W. Steiner and Satyajit Roy
  
  This file is part of nscool_wrap.
  
  nscool_wrap is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 3 of the License, or
  (at your option) any later version.
  
  nscool_wrap is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.
  
  You should have received a copy of the GNU General Public License
  along with O2scl. If not, see <http://www.gnu.org/licenses/>.

  -------------------------------------------------------------------
*/
#ifndef NSCOOL_TOOLS_H
#define NSCOOL_TOOLS_H

#include <iostream>
#include <vector>

class tools {
  public:
  tools(){

  }
  void spline(const std::vector<double>& x, const std::vector<double>& y,
              int in, double yp1, double ypn, std::vector<double>& y2) {
    const int JMAX = 100;
    std::vector<double> u(JMAX);

    if (yp1 >= 1.0e30) {
      y2[0] = 0.0;
      u[0] = 0.0;
    } else {
      y2[0] = -0.5;
      u[0] = (3.0 / (x[1] - x[0])) * ((y[1] - y[0]) / (x[1] - x[0]) - yp1);
    }

    for (int i = 1; i < in - 1; ++i) {
      double sig = (x[i] - x[i - 1]) / (x[i + 1] - x[i - 1]);
      double p = sig * y2[i - 1] + 2.0;
      y2[i] = (sig - 1.0) / p;
      u[i] = (6.0 * ((y[i + 1] - y[i]) / (x[i + 1] - x[i]) - (y[i] - y[i - 1]) / (x[i] - x[i - 1]))
              / (x[i + 1] - x[i - 1]) - sig * u[i - 1]) / p;
    }

    double qn, un;
    std::vector<double> y2n(in);
      
    if (ypn >= 1.0e30) {
      qn = 0.0;
      un = 0.0;
    } else {
      qn = 0.5;
      un = (3.0 / (x[in - 1] - x[in - 2])) * (ypn - (y[in - 1] - y[in - 2]) / (x[in - 1] - x[in - 2]));
    }

    y2n[in - 1] = (un - qn * u[in - 2]) / (qn * y2[in - 2] + 1.0);
      
    for (int k = in - 2; k >= 0; --k) {
      y2n[k] = y2[k] * y2n[k + 1] + u[k];
    }
      
    y2 = y2n;
  }

  void splint(const std::vector<double>& xa, const std::vector<double>& ya,
            const std::vector<double>& y2a, int in, double x, double& y) {
    int klo = 0;
    int khi = in - 1;

    while (khi - klo > 1) {
      int k = (khi + klo) / 2;
      if (xa[k] > x)
        khi = k;
      else
        klo = k;
    }

    double h = xa[khi] - xa[klo];
    double a = (xa[khi] - x) / h;
    double b = (x - xa[klo]) / h;
    y = a * ya[klo] + b * ya[khi] +
      ((a * a * a - a) * y2a[klo] + (b * b * b - b) * y2a[khi]) * (h * h) / 6.0;
  }

  void spline2(const std::vector<double>& x1a, const std::vector<double>& x2a,
              const std::vector<std::vector<double>>& ya,
              int im, int in, std::vector<std::vector<double>>& y2a) {
    std::vector<double> ytmp(in), y2tmp(in);
      
    for (int ij = 0; ij < im; ++ij) {
      for (int ik = 0; ik < in; ++ik) {
        ytmp[ik] = ya[ij][ik];
      }
      spline(x2a, ytmp, in, 1.0e30, 1.0e30, y2tmp);
      for (int ik = 0; ik < in; ++ik) {
        y2a[ij][ik] = y2tmp[ik];
      }
    }
  }

  void splint2(const std::vector<double>& x1a, const std::vector<double>& x2a,
              const std::vector<std::vector<double>>& ya,
              const std::vector<std::vector<double>>& y2a,
              int im, int in, double x1, double x2, double& y) {
    const double BIG = 1.0e30;
      
    std::vector<double> ytmp(in), y2tmp(in), yyytmp(im);
      
    for (int ij = 0; ij < im; ++ij) {
      for (int ik = 0; ik < in; ++ik) {
        ytmp[ik] = ya[ij][ik];
        y2tmp[ik] = y2a[ij][ik];
      }
      splint(x2a, ytmp, y2tmp, in, x2, yyytmp[ij]);
      }
    spline(x1a, yyytmp, im, BIG, BIG, y2tmp);
    splint(x1a, yyytmp, y2tmp, im, x1, y);
  }
};
#endif