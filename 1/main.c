#include <math.h>
#include <omp.h>
#include <stdio.h>

typedef struct _net_t {
    size_t sz;
    double h;

    double** u;
    double** f;
} net_t;

#define BLOCK_SZ 16
#define EPS      0.1

static int min(int a, int b) { return a < b ? a : b; }

static double processBlock(net_t* nt, int a, int b) {
    int i0 = 1 + a * BLOCK_SZ;
    int im = min(i0 + BLOCK_SZ, nt->sz - 1);
    int j0 = 1 + b * BLOCK_SZ;
    int jm = min(j0 + BLOCK_SZ, nt->sz - 1);

    double dm = 0;
    for (int i = i0; i < im; i++) {
        for (int j = j0; j < jm; j++) {
            double temp = nt->u[i][j];
            nt->u[i][j] = 0.25 * (nt->u[i - 1][j] + nt->u[i + 1][j] + nt->u[i][j - 1] + nt->u[i][j + 1] -
                                  nt->h * nt->h * nt->f[i][j]);
            double d = fabs(temp - nt->u[i][j]);
            if (dm < d)
                dm = d;
        }
    }
    return dm;
}

void processNet(net_t* nt) {
    size_t work_sz = nt->sz - 2;
    int numb_block = work_sz / BLOCK_SZ;
    if (BLOCK_SZ * numb_block != work_sz)
        work_sz += 1;
    double dmax = 0;
    double* dm = calloc(work_sz, sizeof(*dm));

    do {
        dmax = 0;
        for (int nx = 0; nx < numb_block; nx++) {
            dm[nx] = 0;
#pragma omp parallel for shared(nt, nx, dm) private(i, j, d)
            for (int i = 0; i < nx + 1; i++) {
                int j = nx - i;
                double d = processBlock(nt, i, j);
                if (dm[i] < d)
                    dm[i] = d;
            } // конец параллельной области
        }
        // затухание волны

        for (int nx = numb_block - 2; nx >= 0; nx--) {
#pragma omp parallel for shared(nt, nx, dm) private(i, j, d)
            for (int i = numb_block - nx - 1; i < numb_block; i++) {
                int j = numb_block + ((numb_block - 2) - nx) - i;
                double d = processBlock(nt, i, j);
                if (dm[i] < d)
                    dm[i] = d;
            } // конец параллельной области
        }

        for (int i = 0; i < numb_block; i++)
            if (dmax < dm[i])
                dmax = dm[i];
        // <определение погрешности вычислений>
    } while (dmax > EPS);
}

int main(int argc, char** argv) {
    printf("So you have a mother!\n");
    return 0;
}