#include <stdio.h>
#include <mpi.h>
#include <math.h>
#include <stdlib.h>
#include <stdbool.h>

#define PRECISION 0.000001
#define RANGESIZE 1000
// #define RANGESIZE 1
#define DATA 0
#define RESULT 1
#define FINISH 2
#define INPUT_FILE "arr2.csv"
#define MAX_ARRAY_SIZE 1000000

// int arr[] = {3, 5, 7, 9, 11, 13, 17, 19};
// sizer = sizeof(arr)/sizeof(arr[0]);

bool isPrime(int n)
{
    if (n < 2)
        return false;
    if (n == 2)
        return true;
    if (n % 2 == 0)
        return false;

    for (int i = 3; i <= sqrt(n); i += 2)
    {
        if (n % i == 0)
        {
            return false;
        }
    }
    return true;
}

// #define DEBUG

bool isTwinPrime(int a, int b)
{
    if (abs(a - b) == 2 && isPrime(a) && isPrime(b))
    {
        return true;
    }
    return false;
}

int countTwinInArray(int *arr, int start, int end)
{
    int count = 0;

    for (int i = start; i < end - 1; i++)
    {
        if (abs(arr[i] - arr[i + 1]) == 2 &&
            isPrime(arr[i]) &&
            isPrime(arr[i + 1]))
        {
            count++;
        }
    }

    return count;
}

int countUniqueTwinPrimesInRange(int a, int b, unsigned char *present, int maxVal)
{
    int count = 0;

    if (a < 2)
        a = 2;

    for (int x = a; x < b; x++)
    {
        if (x + 2 <= maxVal)
        {
            if (present[x] && present[x + 2] &&
                isPrime(x) && isPrime(x + 2))
            {
                count++;
            }
        }
    }
    return count;
}

double f(double x)
{
    return sin(x) * sin(x) / x;
}

// double
// SimpleIntegration (double a, double b)
// {
//     double i;
//     double sum = 0;
//     for (i = a; i < b; i += PRECISION)
//         sum += f(i) * PRECISION;
//     return sum;
// }

int main(int argc, char **argv)
{
    MPI_Request *requests;
    int requestcount = 0;
    int requestcompleted;
    int myrank, proccount;
    double a, b;

    double *ranges;
    double range[2];
    double result = 0;
    double *resulttemp;
    int sentcount = 0;
    int recvcount = 0;
    int i;
    MPI_Status status;

    int n = 0;
    int *inputArray = NULL;
    int maxVal = 0;
    unsigned char *present = NULL;

    // Initialize MPI
    MPI_Init(&argc, &argv);

    // find out my rank
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    // find out the number of processes
    MPI_Comm_size(MPI_COMM_WORLD, &proccount);

    if (proccount < 2)
    {
        printf("Run with at least 2 processes");
        MPI_Finalize();
        return -1;
    }

    if (myrank == 0)
    {
        FILE *file = fopen(INPUT_FILE, "r");
        if (!file)
        {
            printf("Cannot open file\n");
            MPI_Abort(MPI_COMM_WORLD, -1);
        }

        inputArray = (int *)malloc(MAX_ARRAY_SIZE * sizeof(int));

        while (n < MAX_ARRAY_SIZE &&
               fscanf(file, "%d", &inputArray[n]) == 1)
        {
            if (inputArray[n] > maxVal)
                maxVal = inputArray[n];
            n++;
        }

        fclose(file);
    }

    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&maxVal, 1, MPI_INT, 0, MPI_COMM_WORLD);

    present = (unsigned char *)calloc(maxVal + 3, sizeof(unsigned char));

    if (myrank == 0)
    {
        for (i = 0; i < n; i++)
            present[inputArray[i]] = 1;
    }

    MPI_Bcast(present, maxVal + 3, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

    a = 2;
    b = maxVal + 1;

    if (((b - a) / RANGESIZE) < 2 * (proccount - 1))
    {
        printf("More subranges needed");
        MPI_Finalize();
        return -1;
    }

    // MASTER
    if (myrank == 0)
    {
        requests = (MPI_Request *)malloc(3 * (proccount - 1) * sizeof(MPI_Request));

        if (!requests)
        {
            printf("\nNot enough memory");
            MPI_Finalize();
            return -1;
        }

        ranges = (double *)malloc(4 * (proccount - 1) * sizeof(double));
        resulttemp = (double *)malloc((proccount - 1) * sizeof(double));

        range[0] = a;

        for (i = 1; i < proccount; i++)
        {
            range[1] = range[0] + RANGESIZE;
            MPI_Send(range, 2, MPI_DOUBLE, i, DATA, MPI_COMM_WORLD);
            sentcount++;
            range[0] = range[1];
        }

        for (i = 0; i < 2 * (proccount - 1); i++)
            requests[i] = MPI_REQUEST_NULL;

        for (i = 1; i < proccount; i++)
            MPI_Irecv(&(resulttemp[i - 1]), 1, MPI_DOUBLE, i, RESULT,
                      MPI_COMM_WORLD, &(requests[i - 1]));

        for (i = 1; i < proccount; i++)
        {
            range[1] = range[0] + RANGESIZE;
            ranges[2 * i - 2] = range[0];
            ranges[2 * i - 1] = range[1];

            MPI_Isend(&(ranges[2 * i - 2]), 2, MPI_DOUBLE, i, DATA,
                      MPI_COMM_WORLD, &(requests[proccount - 2 + i]));

            sentcount++;
            range[0] = range[1];
        }

        while (range[1] < b)
        {
            MPI_Waitany(2 * proccount - 2, requests,
                        &requestcompleted, MPI_STATUS_IGNORE);

            if (requestcompleted < (proccount - 1))
            {
                result += resulttemp[requestcompleted];
                recvcount++;

                MPI_Wait(&(requests[proccount - 1 + requestcompleted]),
                         MPI_STATUS_IGNORE);

                range[1] = range[0] + RANGESIZE;
                if (range[1] > b)
                    range[1] = b;

                ranges[2 * requestcompleted] = range[0];
                ranges[2 * requestcompleted + 1] = range[1];

                MPI_Isend(&(ranges[2 * requestcompleted]), 2, MPI_DOUBLE,
                          requestcompleted + 1, DATA, MPI_COMM_WORLD,
                          &(requests[proccount - 1 + requestcompleted]));

                sentcount++;
                range[0] = range[1];

                MPI_Irecv(&(resulttemp[requestcompleted]), 1,
                          MPI_DOUBLE, requestcompleted + 1, RESULT,
                          MPI_COMM_WORLD,
                          &(requests[requestcompleted]));
            }
        }

        range[0] = b;
        range[1] = b;

        for (i = 1; i < proccount; i++)
        {
            MPI_Isend(range, 2, MPI_DOUBLE, i, DATA,
                      MPI_COMM_WORLD,
                      &(requests[2 * proccount - 3 + i]));
        }

        MPI_Waitall(3 * proccount - 3, requests, MPI_STATUSES_IGNORE);

        for (i = 0; i < (proccount - 1); i++)
            result += resulttemp[i];

        for (i = 0; i < (proccount - 1); i++)
        {
            MPI_Recv(&(resulttemp[i]), 1, MPI_DOUBLE, i + 1, RESULT,
                     MPI_COMM_WORLD, &status);
            result += resulttemp[i];
        }

        printf("\nUnique twin prime pairs count = %.0f\n", result);
    }
    else // SLAVE
    {
        requests = (MPI_Request *)malloc(2 * sizeof(MPI_Request));

        requests[0] = requests[1] = MPI_REQUEST_NULL;
        ranges = (double *)malloc(2 * sizeof(double));
        resulttemp = (double *)malloc(2 * sizeof(double));

        MPI_Recv(range, 2, MPI_DOUBLE, 0, DATA, MPI_COMM_WORLD, &status);

        while (range[0] < range[1])
        {
            MPI_Irecv(ranges, 2, MPI_DOUBLE, 0, DATA,
                      MPI_COMM_WORLD, &(requests[0]));

            resulttemp[1] = (double)countUniqueTwinPrimesInRange(
                (int)range[0], (int)range[1], present, maxVal);

            MPI_Waitall(2, requests, MPI_STATUSES_IGNORE);

            range[0] = ranges[0];
            range[1] = ranges[1];
            resulttemp[0] = resulttemp[1];

            MPI_Isend(&(resulttemp[0]), 1, MPI_DOUBLE, 0, RESULT,
                      MPI_COMM_WORLD, &(requests[1]));
        }

        MPI_Wait(&(requests[1]), MPI_STATUS_IGNORE);
    }

    MPI_Finalize();
    return 0;
}
