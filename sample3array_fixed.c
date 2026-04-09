#include <stdio.h>
#include <mpi.h>
#include <math.h>
#include <stdlib.h>
#include <stdbool.h>

#define PRECISION 0.000001
#define RANGESIZE 1000
#define DATA 0
#define RESULT 1
#define FINISH 2
#define INPUT_FILE "arr2.csv"
#define MAX_ARRAY_SIZE 2000000

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

// Count unique twin primes in a range where both numbers exist in the array
int countUniqueTwinPrimesInRange(int a, int b, unsigned char *present, int maxVal)
{
    int count = 0;

    if (a < 2)
        a = 2;

    // Check each number x in range [a, b)
    for (int x = a; x < b; x++)
    {
        // For a twin prime pair (x, x+2):
        // Both x and x+2 must be present in the original array
        // AND both must be prime
        if (x + 2 <= maxVal)
        {
            if (present[x] && present[x + 2] &&
                isPrime(x) && isPrime(x + 2))
            {
                // Count this unique twin prime pair (x, x+2)
                count++;
            }
        }
    }
    return count;
}

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
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &proccount);

    if (proccount < 2)
    {
        printf("Run with at least 2 processes\n");
        MPI_Finalize();
        return -1;
    }

    // ============ MASTER PROCESS ============
    if (myrank == 0)
    {
        FILE *file = fopen(INPUT_FILE, "r");
        if (!file)
        {
            printf("Cannot open file: %s\n", INPUT_FILE);
            MPI_Abort(MPI_COMM_WORLD, -1);
        }

        inputArray = (int *)malloc(MAX_ARRAY_SIZE * sizeof(int));
        if (!inputArray)
        {
            printf("Cannot allocate memory for input array\n");
            MPI_Abort(MPI_COMM_WORLD, -1);
        }

        // Read array from file
        while (n < MAX_ARRAY_SIZE && fscanf(file, "%d", &inputArray[n]) == 1)
        {
            if (inputArray[n] > maxVal)
                maxVal = inputArray[n];
            n++;
        }

        fclose(file);
        printf("Loaded %d elements, maxVal = %d\n", n, maxVal);
    }

    // Broadcast array size and max value to all processes
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&maxVal, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Create presence array
    present = (unsigned char *)calloc(maxVal + 3, sizeof(unsigned char));
    if (!present)
    {
        printf("Rank %d: Cannot allocate presence array\n", myrank);
        MPI_Finalize();
        return -1;
    }

    // Mark present elements
    if (myrank == 0)
    {
        for (i = 0; i < n; i++)
        {
            if (inputArray[i] >= 0 && inputArray[i] <= maxVal)
                present[inputArray[i]] = 1;
        }
    }

    // Broadcast presence array
    MPI_Bcast(present, maxVal + 3, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

    // Define search range for twin primes
    a = 2;
    b = maxVal + 1;

    if (((b - a) / RANGESIZE) < 2 * (proccount - 1))
    {
        if (myrank == 0)
            printf("Warning: Not enough subranges for optimal distribution\n");
    }

    // ============ MASTER PROCESS ============
    if (myrank == 0)
    {
        requests = (MPI_Request *)malloc(3 * (proccount - 1) * sizeof(MPI_Request));
        if (!requests)
        {
            printf("Cannot allocate memory for requests\n");
            MPI_Finalize();
            return -1;
        }

        ranges = (double *)malloc(2 * (proccount - 1) * sizeof(double));
        resulttemp = (double *)malloc((proccount - 1) * sizeof(double));

        if (!ranges || !resulttemp)
        {
            printf("Cannot allocate memory\n");
            MPI_Finalize();
            return -1;
        }

        range[0] = a;

        // Send initial ranges to all workers
        for (i = 1; i < proccount; i++)
        {
            range[1] = range[0] + RANGESIZE;
            if (range[1] > b)
                range[1] = b;

            MPI_Send(range, 2, MPI_DOUBLE, i, DATA, MPI_COMM_WORLD);
            sentcount++;
            range[0] = range[1];
        }

        // Prepare for receiving results
        for (i = 0; i < (proccount - 1); i++)
            MPI_Irecv(&(resulttemp[i]), 1, MPI_DOUBLE, i + 1, RESULT,
                      MPI_COMM_WORLD, &(requests[i]));

        // Prepare for sending next ranges
        for (i = 1; i < proccount; i++)
        {
            if (range[0] < b)
            {
                range[1] = range[0] + RANGESIZE;
                if (range[1] > b)
                    range[1] = b;

                ranges[2 * (i - 1)] = range[0];
                ranges[2 * (i - 1) + 1] = range[1];

                MPI_Isend(&(ranges[2 * (i - 1)]), 2, MPI_DOUBLE, i, DATA,
                          MPI_COMM_WORLD, &(requests[proccount - 1 + i - 1]));

                sentcount++;
                range[0] = range[1];
            }
        }

        // Process completed requests
        while (range[0] < b)
        {
            MPI_Waitany(2 * proccount - 2, requests,
                        &requestcompleted, MPI_STATUS_IGNORE);

            if (requestcompleted < (proccount - 1))
            {
                // We received a result
                result += resulttemp[requestcompleted];
                recvcount++;

                // Wait for corresponding send to complete
                MPI_Wait(&(requests[proccount - 1 + requestcompleted]),
                         MPI_STATUS_IGNORE);

                // Send next range if available
                if (range[0] < b)
                {
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
                }

                // Prepare to receive next result
                MPI_Irecv(&(resulttemp[requestcompleted]), 1,
                          MPI_DOUBLE, requestcompleted + 1, RESULT,
                          MPI_COMM_WORLD,
                          &(requests[requestcompleted]));
            }
        }

        // Send termination signal (range[0] == range[1])
        range[0] = b;
        range[1] = b;

        for (i = 1; i < proccount; i++)
        {
            MPI_Send(range, 2, MPI_DOUBLE, i, DATA, MPI_COMM_WORLD);
        }

        // Receive final results from all workers
        for (i = 0; i < (proccount - 1); i++)
        {
            MPI_Recv(&(resulttemp[i]), 1, MPI_DOUBLE, i + 1, RESULT,
                     MPI_COMM_WORLD, &status);
            result += resulttemp[i];
        }

        printf("\n========== RESULTS ==========\n");
        printf("Total unique twin prime pairs count = %.0f\n", result);
        printf("Array size: %d elements\n", n);
        printf("Max value: %d\n", maxVal);
        printf("=============================\n\n");

        // Cleanup
        free(inputArray);
        free(requests);
        free(ranges);
        free(resulttemp);
    }
    else // WORKER PROCESSES
    {
        requests = (MPI_Request *)malloc(2 * sizeof(MPI_Request));
        ranges = (double *)malloc(2 * sizeof(double));
        resulttemp = (double *)malloc(sizeof(double));

        if (!requests || !ranges || !resulttemp)
        {
            printf("Rank %d: Memory allocation failed\n", myrank);
            MPI_Finalize();
            return -1;
        }

        // Receive initial range
        MPI_Recv(range, 2, MPI_DOUBLE, 0, DATA, MPI_COMM_WORLD, &status);

        while (range[0] < range[1])
        {
            // Post receive for next range
            MPI_Irecv(ranges, 2, MPI_DOUBLE, 0, DATA,
                      MPI_COMM_WORLD, &(requests[0]));

            // Process current range
            *resulttemp = (double)countUniqueTwinPrimesInRange(
                (int)range[0], (int)range[1], present, maxVal);

            // Wait for next range to arrive
            MPI_Wait(&(requests[0]), MPI_STATUS_IGNORE);

            // Send result
            MPI_Send(resulttemp, 1, MPI_DOUBLE, 0, RESULT, MPI_COMM_WORLD);

            // Update range for next iteration
            range[0] = ranges[0];
            range[1] = ranges[1];
        }

        // Cleanup
        free(requests);
        free(ranges);
        free(resulttemp);
    }

    // Cleanup shared resources
    free(present);

    MPI_Finalize();
    return 0;
}
