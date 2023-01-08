#include <cmath>

// from “Efficient approximations for the arctangent function”, Rajan, S. Sichun Wang Inkol, R. Joyal, A., May 2006
double fastatan(double x)
{
  return M_PI_4 * x - x * (fabs(x) - 1) * (0.2447 + 0.0663 * fabs(x));
}

double fastatan2(double y, double x)
{
  if (x >= 0)
  { // -pi/2 .. pi/2
    if (y >= 0)
    { // 0 .. pi/2
      if (y < x)
      { // 0 .. pi/4
        return fastatan(y / x);
      }
      else
      { // pi/4 .. pi/2
        return M_PI_2 - fastatan(x / y);
      }
    }
    else
    {
      if (-y < x)
      { // -pi/4 .. 0
        return fastatan(y / x);
      }
      else
      { // -pi/2 .. -pi/4
        return -M_PI_2 - fastatan(x / y);
      }
    }
  }
  else
  { // -pi..-pi/2, pi/2..pi
    if (y >= 0)
    { // pi/2 .. pi
      if (y < -x)
      { // pi*3/4 .. pi
        return fastatan(y / x) + M_PI;
      }
      else
      { // pi/2 .. pi*3/4
        return M_PI_2 - fastatan(x / y);
      }
    }
    else
    { // -pi .. -pi/2
      if (-y < -x)
      { // -pi .. -pi*3/4
        return fastatan(y / x) - M_PI;
      }
      else
      { // -pi*3/4 .. -pi/2
        return -M_PI_2 - fastatan(x / y);
      }
    }
  }
}

