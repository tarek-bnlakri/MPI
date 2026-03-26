#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <time.h>

#define CHUNKSIZE 1000
#define DATA      0
#define RESULT    1
#define FINISH    2

/* Check if a number is prime */
int is_prime(unsigned long n)
{
    if (n < 2) return 0;
    if (n == 2) return 1;
    if (n % 2 == 0) return 0;
    unsigned long limit = (unsigned long)sqrt((double)n);
    for (unsigned long i = 3; i <= limit; i += 2)
        if (n % i == 0) return 0;
    return 1;
}

/* Count twin prime pairs in array[start..start+len-1]
   A twin prime pair is (p, p+2) where both are prime.
   We need to check boundaries with neighbours, so we pass
   the full chunk plus one look-ahead element. */
long count_twins_in_chunk(unsigned long *arr, long start, long len, long total)
{
    long count = 0;
    for (long i = start; i < start + len; i++)
    {
        /* pair (arr[i], arr[i]+2): both must be prime */
        if (is_prime(arr[i]) && is_prime(arr[i] + 2))
            count++;
    }
    return count;
}

int main(int argc, char **argv)
{
    int myrank, proccount;
    MPI_Status status;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &proccount);

    if (proccount < 2)
    {
        printf("Run with at least 2 processes\n");
        MPI_Finalize();
        return -1;
    }

    if (argc < 2)
    {
        if (myrank == 0)
            printf("Usage: %s <array_size>\n", argv[0]);
        MPI_Finalize();
        return -1;
    }

    long array_size = atol(argv[1]);

    if (array_size < 2 * (proccount - 1) * CHUNKSIZE)
    {
        if (myrank == 0)
            printf("Array too small for this many processes, or increase CHUNKSIZE\n");
        MPI_Finalize();
        return -1;
    }

    /* --- MASTER --- */
    if (myrank == 0)
    {
        /* Fill array with random unsigned longs */
        unsigned long *arr = (unsigned long *)malloc(array_size * sizeof(unsigned long));
        if (!arr) { printf("malloc failed\n"); MPI_Finalize(); return -1; }

        srand(42);
        for (long i = 0; i < array_size; i++)
            arr[i] = ((unsigned long)rand() << 16 | rand()) % 1000000 + 2;

        long total_twins = 0;
        long next_start = 0;          /* next chunk start index */
        long chunk_info[2];           /* [start, length] sent to slave */
        long resulttemp;

        /* Send first chunk to each slave */
        for (int i = 1; i < proccount; i++)
        {
            long len = (next_start + CHUNKSIZE <= array_size) ? CHUNKSIZE : array_size - next_start;
            chunk_info[0] = next_start;
            chunk_info[1] = len;
            MPI_Send(arr + next_start, len, MPI_UNSIGNED_LONG, i, DATA, MPI_COMM_WORLD);
            MPI_Send(chunk_info, 2, MPI_LONG, i, DATA, MPI_COMM_WORLD);
            next_start += len;
        }

        /* Dynamic dispatch: whenever a slave finishes, give it the next chunk */
        do
        {
            MPI_Recv(&resulttemp, 1, MPI_LONG, MPI_ANY_SOURCE, RESULT, MPI_COMM_WORLD, &status);
            total_twins += resulttemp;

            if (next_start < array_size)
            {
                long len = (next_start + CHUNKSIZE <= array_size) ? CHUNKSIZE : array_size - next_start;
                chunk_info[0] = next_start;
                chunk_info[1] = len;
                MPI_Send(arr + next_start, len, MPI_UNSIGNED_LONG, status.MPI_SOURCE, DATA, MPI_COMM_WORLD);
                MPI_Send(chunk_info, 2, MPI_LONG, status.MPI_SOURCE, DATA, MPI_COMM_WORLD);
                next_start += len;
            }
            else
            {
                /* No more work for this slave — send FINISH */
                MPI_Send(NULL, 0, MPI_UNSIGNED_LONG, status.MPI_SOURCE, FINISH, MPI_COMM_WORLD);
            }
        }
        while (next_start < array_size);

        /* Collect remaining in-flight results */
        for (int i = 1; i < proccount; i++)
        {
            MPI_Recv(&resulttemp, 1, MPI_LONG, MPI_ANY_SOURCE, RESULT, MPI_COMM_WORLD, &status);
            total_twins += resulttemp;
            MPI_Send(NULL, 0, MPI_UNSIGNED_LONG, status.MPI_SOURCE, FINISH, MPI_COMM_WORLD);
        }

        printf("\nArray size: %ld\n", array_size);
        printf("Twin prime pairs found: %ld\n", total_twins);

        free(arr);
    }
    /* --- SLAVE --- */
    else
    {
        unsigned long *chunk = (unsigned long *)malloc(CHUNKSIZE * sizeof(unsigned long));
        long chunk_info[2];

        do
        {
            MPI_Probe(0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

            if (status.MPI_TAG == DATA)
            {
                int count;
                MPI_Get_count(&status, MPI_UNSIGNED_LONG, &count);
                MPI_Recv(chunk, count, MPI_UNSIGNED_LONG, 0, DATA, MPI_COMM_WORLD, &status);
                MPI_Recv(chunk_info, 2, MPI_LONG, 0, DATA, MPI_COMM_WORLD, &status);

                long result = count_twins_in_chunk(chunk, 0, count, array_size);
                MPI_Send(&result, 1, MPI_LONG, 0, RESULT, MPI_COMM_WORLD);
            }
        }
        while (status.MPI_TAG != FINISH);

        free(chunk);
    }

    MPI_Finalize();
    return 0;
}
