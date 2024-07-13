
#include <stdio.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include "rt_nonfinite.h"

double *myDoubleArray = NULL;

void filterScreen(double b[2], double a[2], const double x[56], double zi, double y[56])
{
    int k;
    int naxpy;
    int j;
    double as;
    if ((!rtIsInf(a[0])) && (!rtIsNaN(a[0])) && (!(a[0] == 0.0)) && (a[0] != 1.0))
    {
        for (k = 0; k < 2; k++)
        {
            b[k] /= a[0];
        }

        a[1] /= a[0];
    }

    y[0] = zi;
    memset(&y[1], 0, 55U * sizeof(double));
    for (k = 0; k < 56; k++)
    {
        naxpy = 56 - k;
        if (naxpy < 2)
        {
            naxpy = 1;
        }
        else
        {
            naxpy = 2;
        }

        for (j = 0; j + 1 <= naxpy; j++)
        {
            y[k + j] += x[k] * b[j];
        }

        naxpy = 55 - k;
        naxpy = !(naxpy < 1);
        as = -y[k];
        j = 1;
        while (j <= naxpy)
        {
            y[k + 1] += as * a[1];
            j = 2;
        }
    }
}

/* Function Definitions */
void remove_baseline(const double ecg[50])
{
    printf("i was called\n");
    int n = 50;
    double xtmp;
    double d0;
    int i;
    double y[56];
    double b_y[56];
    double dv0[2];
    double dv1[2];

    myDoubleArray = (double *)malloc(n * sizeof(double));

    static const double dv2[2] = {1.0, -0.924390491658207};

    /*  Sampling frequency (Hz) */
    /* made sure that removing baseline wandering periodically will also reflect */
    /* for complete array. */
    /*  Cutoff frequency (Hz) */
    /*  Apply the filter to the ECG signal */
    xtmp = 2.0 * ecg[0];
    d0 = 2.0 * ecg[49];
    for (i = 0; i < 3; i++)
    {
        y[i] = xtmp - ecg[3 - i];
    }

    memcpy(&y[3], &ecg[0], 50U * sizeof(double));
    for (i = 0; i < 3; i++)
    {
        y[i + 53] = d0 - ecg[48 - i];
    }

    for (i = 0; i < 2; i++)
    {
        dv0[i] = 0.96219524582910343 + -1.9243904916582069 * (double)i;
        dv1[i] = dv2[i];
    }

    memcpy(&b_y[0], &y[0], 56U * sizeof(double));
    filterScreen(dv0, dv1, b_y, -0.96219524582910365 * y[0], y);
    for (i = 0; i < 28; i++)
    {
        xtmp = y[i];
        y[i] = y[55 - i];
        y[55 - i] = xtmp;
    }

    for (i = 0; i < 2; i++)
    {
        dv0[i] = 0.96219524582910343 + -1.9243904916582069 * (double)i;
        dv1[i] = dv2[i];
    }

    memcpy(&b_y[0], &y[0], 56U * sizeof(double));
    filterScreen(dv0, dv1, b_y, -0.96219524582910365 * y[0], y);
    for (i = 0; i < 28; i++)
    {
        xtmp = y[i];
        y[i] = y[55 - i];
        y[55 - i] = xtmp;
    }

    memcpy(&myDoubleArray[0], &y[3], 50U * sizeof(double));
}

int main()
{
    double ecg[50] = {717, 727, 714, 696, 676, 672, 697, 718, 729, 712, 706, 693, 679, 693, 718, 724, 727, 720, 690, 696, 708, 718, 715, 703, 712, 726, 723, 694, 706, 720, 717, 709, 708, 714, 726, 723, 726, 723, 733, 733, 718, 705, 720, 733, 729, 726, 742, 738, 729, 729};
    remove_baseline(ecg);

    printf("\n After");
    for (int i = 0; i < 50; i++)
    {
        printf("%f,", myDoubleArray[i]);
    }
}
