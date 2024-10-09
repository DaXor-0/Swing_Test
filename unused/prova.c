#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>

int int_pow(int base, int exp) {
  int result = 1;
  while (exp > 0) {
    if (exp % 2 == 1) result *= base;
    base *= base;
    exp /= 2;
  }
  return result;
}

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);

  int my_rank, comm_sz;
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
  
  int what_to_print = 3; // 1 = SOURCE, 2 = RESULT, 3 = ONLY ERRORS, 4 = SOURCE AND RESULT, anything else is NOTHING
  int dim = 8;
  switch (argc){
    case 2:
      {
        dim = atoi(argv[1]);
        break;
      }
    case 3:
      {
        dim = atoi(argv[1]);
        what_to_print = atoi(argv[2]);
        break;
      }
    default:
      {
        break;
      }
  }

  int sendbuf[dim], recvbuf[dim];
  
  for(int i=0; i<dim; i++){
    sendbuf[i] = int_pow(10, my_rank);
  }

  MPI_Allreduce(sendbuf, recvbuf, dim, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  
  if(what_to_print == 1){
    //PRINT SOURCEBUF
    for (int rank = 0; rank < comm_sz; rank++) {
      if (my_rank == rank) {
        printf("\nRANK %d\n", my_rank);
        for (int ind = 0; ind < dim; ind++){
          printf("%d\t",sendbuf[ind]);
        }
        printf("\n");
        fflush(stdout);
      }
      MPI_Barrier(MPI_COMM_WORLD);
    }
  }
  else if(what_to_print == 2){
    //PRINT RESULT OF ALLREDUCE
    for (int rank = 0; rank < comm_sz; rank++) {
      if (my_rank == rank) {
        printf("\nRANK %d\n", my_rank);
        for (int ind = 0; ind < dim; ind++){
          printf("%d\t",recvbuf[ind]);
        }
        printf("\n");
        fflush(stdout);
      }
      MPI_Barrier(MPI_COMM_WORLD);
    }
  }
  else if(what_to_print == 3){
    //PRINT ONLY WRONG RESULTS
    int expected = 0;
    for (int i = 0; i < comm_sz; i++){
      expected += int_pow(10, i);
    }
    int flag = 1;
    for (int rank = 0; rank < comm_sz; rank++) {
      if (my_rank == rank) {
        for (int ind = 0; ind < dim; ind++){
          if (recvbuf[ind] != expected){
            if (flag){
              printf("\nRANK %d\n", my_rank);
              flag = 0;
            }
            printf("%d:%d\t", ind, recvbuf[ind]);
          }
        }
        if(flag == 0){
          printf("\n");
          fflush(stdout);
        }
      }
      MPI_Barrier(MPI_COMM_WORLD);
    }
  }
  else if(what_to_print == 4){
    //PRINT SOURCE AND RESULT
    for (int rank = 0; rank < comm_sz; rank++) {
      if (my_rank == rank) {
        printf("\nRANK %d\nsrc\tres\n", my_rank);
        for (int ind = 0; ind < dim; ind++){
          printf("%d\t%d\n", sendbuf[ind], recvbuf[ind]);
        }
        printf("\n");
        fflush(stdout);
      }
      MPI_Barrier(MPI_COMM_WORLD);
    }
  }

  MPI_Finalize();
  return 0;
}
