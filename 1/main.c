#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct _net_t {
    size_t sz;
    double h;

    double** u;
    double** f;
} net_t;

typedef struct _test_res_t {
    size_t iter;
    double t;
} test_res_t;

#define BLOCK_SZ 64
#define EPS      0.1

typedef double (*fun_xy)(double, double);

double d_x3_p_y3(double x, double y) { return 6 * x + 6 * y; }

double x3_p_y3(double x, double y) { return pow(x, 3) + pow(y, 3); }

double d_kx3_p_2ky3(double x, double y) { return 6000 * x + 12000 * y; }

double kx3_p_2ky3(double x, double y) { return 1000 * pow(x, 3) + 2000 * pow(y, 3); }

double d_book(double x, double y) { return 0; }

double book(double x, double y) { return (1 - 2 * y) * (1 - 2 * x) * 100; }

double** create_double_2d_arr(size_t sz) {
    double** res = calloc(sz, sizeof(*res));
    for (int i = 0; i < sz; i++)
        res[i] = calloc(sz, sizeof(*res[i]));
    return res;
}

void free_double_2d_arr(double** arr, size_t sz) {
    for (int i = 0; i < sz; i++)
        free(arr[i]);
    return free(arr);
}

net_t* create_net_t(size_t sz, fun_xy f, fun_xy u) {
    net_t* res = malloc(sizeof(*res));
    res->sz = sz;
    res->h = 1.0 / (sz - 1);
    res->u = create_double_2d_arr(sz);
    res->f = create_double_2d_arr(sz);

    for (int i = 0; i < sz; i++) {
        for (int j = 0; j < sz; j++) {
            if ((i == 0) || (j == 0) || (i == (sz - 1)) || (j == (sz - 1))) {
                res->u[i][j] = u(i * res->h, j * res->h);
            } else {
                res->u[i][j] = 0;
            }
            res->f[i][j] = f(i * res->h, j * res->h);
        }
    }
    return res;
}

void free_net_t(net_t* nt) {
    free_double_2d_arr(nt->u, nt->sz);
    free_double_2d_arr(nt->f, nt->sz);
    return free(nt);
}

void print_tb(double** tb, size_t sz) {
    for (int i = 0; i < sz; i++) {
        for (int j = 0; j < sz; j++) {
            printf("%7.2f ", tb[i][j]);
        }
        printf("\n");
    }
}

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

size_t processNet(net_t* nt) {
    size_t iter = 0;
    size_t work_sz = nt->sz - 2;
    int numb_block = work_sz / BLOCK_SZ;
    if (BLOCK_SZ * numb_block != work_sz)
        numb_block += 1;
    double dmax = 0;
    double* dm = calloc(numb_block, sizeof(*dm));

    do {
        iter++;
        dmax = 0;
        for (int nx = 0; nx < numb_block; nx++) {
            dm[nx] = 0;

            int i, j;
            double d;

#pragma omp parallel for shared(nt, nx, dm) private(i, j, d)
            for (i = 0; i < nx + 1; i++) {
                j = nx - i;
                d = processBlock(nt, i, j);
                if (dm[i] < d)
                    dm[i] = d;
            } // конец параллельной области
        }
        // затухание волны

        for (int nx = numb_block - 2; nx >= 0; nx--) {
            int i, j;
            double d;

#pragma omp parallel for shared(nt, nx, dm) private(i, j, d)
            for (i = numb_block - nx - 1; i < numb_block; i++) {
                j = numb_block + ((numb_block - 2) - nx) - i;
                d = processBlock(nt, i, j);
                if (dm[i] < d)
                    dm[i] = d;
            } // конец параллельной области
        }

        for (int i = 0; i < numb_block; i++)
            if (dmax < dm[i])
                dmax = dm[i];
        // <определение погрешности вычислений>
    } while (dmax > EPS);
    free(dm);

    return iter;
}

test_res_t run_test(size_t sz, int threads_num, fun_xy f, fun_xy u) {
    omp_set_num_threads(threads_num);
    net_t* nt = create_net_t(sz, f, u);
    // printf("\n####### Init ########\n");
    // print_tb(nt->u, nt->sz);

    double t1, t2, dt;
    t1 = omp_get_wtime();
    size_t iter = processNet(nt);
    t2 = omp_get_wtime();
    dt = t2 - t1;

    // printf("\n####### Result ########\n");
    // print_tb(nt->u, nt->sz);

    // net_t* ntcheck = create_net_t(sz, u, u);
    // printf("\n####### Real value ###########\n");
    // print_tb(ntcheck->f, ntcheck->sz);

    // double mx = -INFINITY;
    // double mn = INFINITY;
    // double sm = 0;

    // for (int i = 0; i < nt->sz; i++) {
    //     for (int j = 0; j < nt->sz; j++) {
    //         ntcheck->u[i][j] = ntcheck->f[i][j] - nt->u[i][j];
    //         sm += fabs(ntcheck->u[i][j]);
    //         mx = fmax(mx, ntcheck->u[i][j]);
    //         mn = fmin(mn, ntcheck->u[i][j]);
    //     }
    // }

    // printf("\n####### Difference ###########\n");
    // print_tb(ntcheck->u, ntcheck->sz);
    // printf("min = %f | max = %f \n", mn, mx);
    // printf("avr = %f \n", sm / (nt->sz * nt->sz));

    free_net_t(nt);

    test_res_t res;
    res.iter = iter;
    res.t = dt;
    return res;
}

int main(int argc, char** argv) {
    fun_xy f = d_kx3_p_2ky3; // d_book;
    fun_xy u = kx3_p_2ky3;   // book;

    size_t sz[] = {100, 200, 300, 500, 1000, 2000, 3000};
    int threads[] = {1, 8};

    size_t lsz = sizeof(sz) / sizeof(sz[0]);
    int lthreads = sizeof(threads) / sizeof(threads[0]);

    for (int i = 0; i < lthreads; i++) {
        int thr = threads[i];
        for (int j = 0; j < lsz; j++) {
            size_t s = sz[j];
            test_res_t tm = run_test(s, thr, f, u);
            printf("Res for sz = %4lu, threads = %d: %10lu | %7.2f s. \n", s, thr, tm.iter, tm.t);
        }
    }

    return 0;
}