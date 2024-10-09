#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>

static inline int opal_hibit(int value, int start)
{
    unsigned int mask;

    --start;
    mask = 1 << start;

    for (; start >= 0; --start, mask >>= 1) {
        if (value & mask) {
            break;
        }
    }

    return start;
}


static inline int pow_of_neg_two(int n) {
  int power_of_two = 1 << n;
  // If n is even, return 2^n, otherwise return -2^n
  return (n % 2 == 0) ? power_of_two : -power_of_two;
}

static inline int pi(int r, int s, int p) {
  int rho_s = (1 - pow_of_neg_two(s + 1)) / 3;
  int result;
  if (r % 2 == 0) result = (r + rho_s) % p;
  else            result = (r - rho_s) % p;

  if (result < 0) result += p;

  return result;
}


static inline void get_indexes_aux(int rank, int step, const int n_steps, const int adj_size, unsigned char *bitmap){
  if (step >= n_steps) return;

  int peer;
  
  for (int s = step; s < n_steps; s++){
    peer = pi(rank, s, adj_size);
    *(bitmap + peer) = 0x1;
    get_indexes_aux(peer, s + 1, n_steps, adj_size, bitmap);
  }

}


static inline void get_indexes(int rank, int step, const int n_steps, const int adj_size, unsigned char *bitmap){
  if (step >= n_steps) return;
  
  int peer = pi(rank, step, adj_size);
  *(bitmap + peer) = 0x1;
  get_indexes_aux(peer, step + 1, n_steps, adj_size, bitmap);
}


int main (int argc, char** argv){
  int comm_sz = 4, bitmap_offset;
  
  switch (argc){
    case (2):
      {
        comm_sz = atoi(argv[1]);
        break;
      }
    default:
      {
        break;
      }
  }

  int max_int_pos = (int) (sizeof(comm_sz) * 8) - 1;
  int n_steps = opal_hibit(comm_sz, max_int_pos);
  
  int adj_size = 1 << n_steps;
  
  printf("ADJ_SIZE %d N_STEPS %d\n\n", adj_size, n_steps);

  unsigned char* s_bitmap = (unsigned char*) calloc(adj_size * n_steps, sizeof(unsigned char));
  unsigned char* r_bitmap = (unsigned char*) calloc(adj_size * n_steps, sizeof(unsigned char));

  for (int rank = 0; rank < adj_size; rank++){
    bitmap_offset = 0;
    printf("-----RANK %d-----\nInternal print:\n", rank);
    for (int step = 0; step < n_steps; step++){
      int dest = pi(rank, step, adj_size);
      get_indexes(rank, step, n_steps, adj_size, s_bitmap + bitmap_offset);
      get_indexes(dest, step, n_steps, adj_size, r_bitmap + bitmap_offset);
      printf("step %d bitmap_offset % d\nsend ", step, bitmap_offset);
      for(int i = 0; i < adj_size; i++)
        printf("%u,", s_bitmap[i + bitmap_offset]);
      printf("recv ");
      for(int i = 0; i < adj_size; i++)
        printf("%u,", r_bitmap[i+ bitmap_offset]);
      printf("\n");
      bitmap_offset += adj_size;
    }
    printf("\nFinal print:\nsend ");
    for(int i = 0; i < adj_size * n_steps; i++)
      printf("%u,", s_bitmap[i]);
    printf("\nrecv ");
    for(int i = 0; i < adj_size * n_steps; i++)
      printf("%u,", r_bitmap[i]);
    printf("\n\n\n");
    memset(s_bitmap, 0, adj_size * n_steps * sizeof(unsigned char));
    memset(r_bitmap, 0, adj_size * n_steps * sizeof(unsigned char));
  }

  free(s_bitmap);
  free(r_bitmap);

}


