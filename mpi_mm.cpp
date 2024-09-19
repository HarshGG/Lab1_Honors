/******************************************************************************
* FILE: mpi_mm.c
* DESCRIPTION:  
*   MPI Matrix Multiply - C Version
*   In this code, the master task distributes a matrix multiply
*   operation to numtasks-1 worker tasks.
* AUTHOR: Blaise Barney. Adapted from Ros Leibensperger, Cornell Theory
*   Center. Converted to MPI: George L. Gusciora, MHPCC (1/95)
* LAST REVISED: 09/29/2021
******************************************************************************/
#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>

#define MASTER 0               /* taskid of first task */
#define FROM_MASTER 1          /* setting a message type */
#define FROM_WORKER 2          /* setting a message type */

int main(int argc, char *argv[])
{
    int sizeOfMatrix;
    if (argc == 2)
    {
        sizeOfMatrix = atoi(argv[1]);
    }
    else
    {
        printf("\n Please provide the size of the matrix.\n");
        return 0;
    }

    int numtasks, taskid, numworkers, rows, averow, extra, offset, i, j, k;
    MPI_Status status;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &taskid);
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);

    if (numtasks < 2) {
        printf("Need at least two MPI tasks. Quitting...\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
        exit(1);
    }

    double a[sizeOfMatrix][sizeOfMatrix], b[sizeOfMatrix][sizeOfMatrix], c[sizeOfMatrix][sizeOfMatrix];
    double start_time, end_time, total_time, send_time, recv_time, comp_time;
    double min_time, max_time, avg_time;
    numworkers = numtasks - 1;

    /**************************** master task ************************************/
    if (taskid == MASTER)
    {
        printf("mpi_mm has started with %d tasks.\n", numtasks);
        printf("Initializing arrays...\n");
        start_time = MPI_Wtime();
        for (i = 0; i < sizeOfMatrix; i++)
            for (j = 0; j < sizeOfMatrix; j++)
                a[i][j] = i + j;
        for (i = 0; i < sizeOfMatrix; i++)
            for (j = 0; j < sizeOfMatrix; j++)
                b[i][j] = i * j;
        end_time = MPI_Wtime();
        total_time = end_time - start_time; // Time for initialization

        printf("Initialization time: %f seconds.\n", total_time);

        // Distribution of tasks to workers
        averow = sizeOfMatrix / numworkers;
        extra = sizeOfMatrix % numworkers;
        offset = 0;
        start_time = MPI_Wtime();
        for (int dest = 1; dest <= numworkers; dest++) {
            rows = (dest <= extra) ? averow + 1 : averow;
            MPI_Send(&offset, 1, MPI_INT, dest, FROM_MASTER, MPI_COMM_WORLD);
            MPI_Send(&rows, 1, MPI_INT, dest, FROM_MASTER, MPI_COMM_WORLD);
            MPI_Send(&a[offset][0], rows * sizeOfMatrix, MPI_DOUBLE, dest, FROM_MASTER, MPI_COMM_WORLD);
            MPI_Send(&b, sizeOfMatrix * sizeOfMatrix, MPI_DOUBLE, dest, FROM_MASTER, MPI_COMM_WORLD);
            offset += rows;
        }
        end_time = MPI_Wtime();
        send_time = end_time - start_time; // Time for sending data

        // Receiving results from worker tasks
        start_time = MPI_Wtime();
        for (i = 1; i <= numworkers; i++) {
            MPI_Recv(&offset, 1, MPI_INT, MPI_ANY_SOURCE, FROM_WORKER, MPI_COMM_WORLD, &status);
            MPI_Recv(&rows, 1, MPI_INT, status.MPI_SOURCE, FROM_WORKER, MPI_COMM_WORLD, &status);
            MPI_Recv(&c[offset][0], rows * sizeOfMatrix, MPI_DOUBLE, status.MPI_SOURCE, FROM_WORKER, MPI_COMM_WORLD, &status);
        }
        end_time = MPI_Wtime();
        recv_time = end_time - start_time; // Time for receiving data

        // Using MPI_Reduce to gather minimum, maximum, and average times
        MPI_Reduce(&send_time, &min_time, 1, MPI_DOUBLE, MPI_MIN, MASTER, MPI_COMM_WORLD);
        MPI_Reduce(&send_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, MASTER, MPI_COMM_WORLD);
        MPI_Reduce(&send_time, &avg_time, 1, MPI_DOUBLE, MPI_SUM, MASTER, MPI_COMM_WORLD);
        avg_time /= numworkers; // Calculate average

        printf("Min Send Time: %f, Max Send Time: %f, Avg Send Time: %f\n", min_time, max_time, avg_time);
    }
    else
    {
        /**************************** worker task ************************************/
        // Receiving data
        MPI_Recv(&offset, 1, MPI_INT, MASTER, FROM_MASTER, MPI_COMM_WORLD, &status);
        MPI_Recv(&rows, 1, MPI_INT, MASTER, FROM_MASTER, MPI_COMM_WORLD, &status);
        MPI_Recv(&a, rows * sizeOfMatrix, MPI_DOUBLE, MASTER, FROM_MASTER, MPI_COMM_WORLD, &status);
        MPI_Recv(&b, sizeOfMatrix * sizeOfMatrix, MPI_DOUBLE, MASTER, FROM_MASTER, MPI_COMM_WORLD, &status);

        // Matrix multiplication computation
        start_time = MPI_Wtime();
        for (k = 0; k < sizeOfMatrix; k++)
            for (i = 0; i < rows; i++) {
                c[i][k] = 0.0;
                for (j = 0; j < sizeOfMatrix; j++)
                    c[i][k] += a[i][j] * b[j][k];
            }
        end_time = MPI_Wtime();
        comp_time = end_time - start_time; // Time for computation

        // Sending data back to master
        MPI_Send(&offset, 1, MPI_INT, MASTER, FROM_WORKER, MPI_COMM_WORLD);
        MPI_Send(&rows, 1, MPI_INT, MASTER, FROM_WORKER, MPI_COMM_WORLD);
        MPI_Send(&c, rows * sizeOfMatrix, MPI_DOUBLE, MASTER, FROM_WORKER, MPI_COMM_WORLD);

        // Using MPI_Reduce to send times to master for min, max, average calculations
        MPI_Reduce(&comp_time, &min_time, 1, MPI_DOUBLE, MPI_MIN, MASTER, MPI_COMM_WORLD);
        MPI_Reduce(&comp_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, MASTER, MPI_COMM_WORLD);
        MPI_Reduce(&comp_time, &avg_time, 1, MPI_DOUBLE, MPI_SUM, MASTER, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}
