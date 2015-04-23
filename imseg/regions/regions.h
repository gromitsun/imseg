#include <stddef.h>
#define MP 1

void cinit_average(double * arr, ssize_t size, double * thresholds, int nthresholds, double * ave)
{
    double region_values[nthresholds + 1];
    ssize_t region_count[nthresholds + 1];
    #pragma omp parallel for if(MP)
    for (ssize_t i = 0; i < size; i++)
    {
        if (arr[i] <= thresholds[0])
        {
            region_values[0] += arr[i];
            region_count[0]++;
            ave[i] = 0;
        }
        else if (arr[i] > thresholds[nthresholds - 1])
        {
            region_values[nthresholds] += arr[i];
            region_count[nthresholds]++;
            ave[i] = nthresholds;
        }
        else
        {
            for (int j = 1; j < nthresholds; j++)
                if ((arr[i] > thresholds[j - 1]) && (arr[i] <= thresholds[j]))
                {
                    region_values[j] += arr[i];
                    region_count[j]++;
                    ave[i] = j;
                    break;
                }
        }
    }

    for (int i = 0; i < nthresholds + 1; i++)
        region_values[i] /= region_count[i];
    #pragma omp parallel for if(MP)
    for (ssize_t i = 0; i < size; i++)
        ave[i] = region_values[(int) ave[i]];
}