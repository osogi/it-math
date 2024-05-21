#include <math.h>
#include <stdio.h>
#include <stdlib.h>

typedef double (*fun_x_t)(double);

typedef struct _thomas_solver_t {
    size_t n;
    double* a;
    double* b;
    double* c;
    double* d;

} thomas_solver_t;

double* run_thomas(thomas_solver_t* solver, double* y) {
    double* a = solver->a;
    double* b = solver->b;
    double* c = solver->c;
    double* d = solver->d;
    size_t n = solver->n;

    for (int i = 2; i <= n; i++) {
        int j = i - 1;
        double w = a[j] / b[j - 1];
        b[j] = b[j] - w * c[j - 1];
        d[j] = d[j] - w * d[j - 1];
    }

    y[n - 1] = d[n - 1] / b[n - 1];
    for (int i = n - 1; i >= 1; i--) {
        int j = i - 1;
        y[j] = (d[j] - c[j] * y[j + 1]) / b[j];
    }
    return y;
}

typedef struct _solution_t {
    double* x;
    double* y;
    double h;

    size_t n;
} solution_t;

solution_t* create_solution_t(double* x, double* y, size_t n) {
    solution_t* solve = malloc(sizeof(*solve));
    solve->x = x;
    solve->y = y;
    solve->n = n;

    solve->h = (x[n] - x[0]) / n;
    return solve;
}

void free_solution_t(solution_t* s, char del_xs) {
    if (del_xs)
        free(s->x);
    free(s->y);
    free(s);
}

double phi(solution_t* s, size_t i, double x) {
    if (i == 0) {
        if ((s->x[0] <= x) && (x <= s->x[1]))
            return (s->x[1] - x) / s->h;
        else
            return 0;
    } else if (i == s->n) {
        if ((s->x[s->n - 1] <= x) && (x <= s->x[s->n]))
            return (x - s->x[s->n - 1]) / s->h;
        else
            return 0;
    } else {
        if ((s->x[i - 1] <= x) && (x <= s->x[i]))
            return (x - s->x[i - 1]) / s->h;
        else if ((s->x[i] <= x) && (x <= s->x[i + 1]))
            return (s->x[i + 1] - x) / s->h;
        else
            return 0;
    }
}

double function(solution_t* s, double x) {
    size_t l = 0;
    size_t r = s->n;

    while (r - l > 1) {
        size_t mid = (l + r) / 2;
        if (x > s->x[mid]) {
            l = mid;
        } else {
            r = mid;
        }
    }
    double yl = s->y[l];
    double pl = phi(s, l, x);
    double yr = s->y[r];
    double pr = phi(s, r, x);
    double res = yl * pl + yr * pr;
    // printf("##############33\n%f * %f + \n%f * %f = \n%f\n", yl, pl, yr, pr, res);

    return res;
}

/*

This FEM solver fits only for the following equation
y'' - λ*y = -2*λ * sin(sqrt(λ)*x)

Решение:
y = sin(sqrt(λ)*x)
Тогда:
p = 1
q = λ
f = 2*λ * sin(sqrt(λ)*x)


*/

// This FEM solver fits only for the following equation
// y'' - λ*y = -2*λ * sin(sqrt(λ)*x)
// y(0) = y(l) = 0
typedef struct _fem_solver_t {
    double lambda;
    size_t N;
    double h;
    double* x;

} fem_solver_t;

fem_solver_t* create_fem_solver_t(double lambda, double l, size_t grid_size) {
    fem_solver_t* solver = malloc(sizeof(*solver));
    solver->lambda = lambda;
    solver->N = grid_size - 1;

    solver->h = l / (grid_size - 1);
    solver->x = calloc(grid_size, sizeof(*(solver->x)));
    for (int i = 0; i < grid_size; i++) {
        solver->x[i] = solver->h * i;
    }
    solver->x[solver->N] = l; // just reinsurance

    return solver;
}

void free_fem_solver_t(fem_solver_t* s, char del_xs) {
    if (del_xs)
        free(s->x);
    free(s);
}

double dot_metric_A_phi(fem_solver_t* solv, int i, int j) {
    if (i > j) {
        int buf = i;
        i = j;
        j = buf;
    }

    double h = solv->h;
    double lambda = solv->lambda;
    double xim = solv->x[i - 1];
    double xi = solv->x[i];
    double xip = solv->x[i + 1];

    if (i == j) {
        // https://www.wolframalpha.com/input?i=%281%2F%28i%5E2%29%29+*%28%28%28integrate+%5B1%2Ba*%28x-m%29%5E2%5D+dx+from+m+to+b%29%29%2B+%28integrate+%5B1%2Ba*%28c-x%29%5E2%5D+dx+from+b+to+c%29%29&assumption=%22i%22+-%3E+%22Variable%22
        // (c + a b^2 c - a b c^2 + (a c^3)/3 - m - a b^2 m + a b m^2 - (a m^3)/3)/i^2
        // i = h
        // a = λ
        // m = x[i-1]
        // b = x[i]
        // c = x[i+1]
        return (lambda * pow(xi, 2) * xip - lambda * xi * pow(xip, 2) + (lambda * pow(xip, 3)) / 3 -
                lambda * pow(xi, 2) * xim + lambda * xi * pow(xim, 2) - (lambda * pow(xim, 3)) / 3 + xip - xim) /
               pow(h, 2);

    } else if (i + 1 == j) {
        // https://www.wolframalpha.com/input?i=%281%2F%28i%5E2%29%29+*+%28integrate+%5B-1%2Ba*%28x-b%29*%28c-x%29%5D+dx+from+b+to+c%29&assumption=%22i%22+-%3E+%22Variable%22
        // (b - (a b^3)/6 - c + 1/2 a b^2 c - 1/2 a b c^2 + (a c^3)/6)/i^2
        // i = h
        // a = λ
        // b = x[i]
        // c = x[i+1]
        return (-(lambda * pow(xi, 3)) / 6 + 1 / 2 * lambda * pow(xi, 2) * xip - 1 / 2 * lambda * xi * pow(xip, 2) +
                (lambda * pow(xip, 3)) / 6 + xi - xip) /
               pow(h, 2);
    } else {
        return 0;
    }
}

double dot_metric_f_and_phi(fem_solver_t* solv, int i) {
    double h = solv->h;
    double lambda = solv->lambda;
    double xim = solv->x[i - 1];
    double xi = solv->x[i];
    double xip = solv->x[i + 1];

    // https://www.wolframalpha.com/input?i=%281%2F%28i%5E2%29%29+*%28%28%28integrate+%5B%28x-m%29*%282*a+*+sin%28sqrt%28a%29*x%29%29%5D+dx+from+m+to+b%29%29%2B+%28integrate+%5B%28c-x%29*%282*a*+sin%28sqrt%28a%29*x%29%29%5D+dx+from+b+to+c%29%29&assumption=%22i%22+-%3E+%22Variable%22
    /*
    (2 (-sqrt(a) (b - c) cos(sqrt(a) b) + sin(sqrt(a) b) - sin(sqrt(a) c)) + 2 (-sqrt(a) (b - m) cos(sqrt(a) b) +
    sin(sqrt(a) b) - sin(sqrt(a) m)))/i^2
    */
    // i = h
    // a = λ
    // m = x[i-1]
    // b = x[i]
    // c = x[i+1]
    return (2 * (-(xi - xip) * sqrt(lambda) * cos(xi * sqrt(lambda)) + sin(xi * sqrt(lambda)) -
                 sin(xip * sqrt(lambda))) +
            2 * (-sqrt(lambda) * (xi - xim) * cos(xi * sqrt(lambda)) + sin(xi * sqrt(lambda)) -
                 sin(sqrt(lambda) * xim))) /
           pow(h, 1);
}

solution_t* run_fem(fem_solver_t* s) {
    thomas_solver_t ts;
    ts.n = s->N - 1;
    ts.a = calloc(ts.n, sizeof(double));
    ts.b = calloc(ts.n, sizeof(double));
    ts.c = calloc(ts.n, sizeof(double));
    ts.d = calloc(ts.n, sizeof(double));

    for (int i = 1; i <= s->N - 1; i++) {
        int j = i - 1;

        if (i - 1 >= 1)
            ts.a[j] = dot_metric_A_phi(s, i - 1, i);

        if (i + 1 < s->N)
            ts.c[j] = dot_metric_A_phi(s, i + 1, i);

        ts.b[j] = dot_metric_A_phi(s, i, i);
        ts.d[j] = dot_metric_f_and_phi(s, i);

        // printf("[%d]: ###############\n", i);
        // printf("a = %f\n", ts.a[j]);
        // printf("b = %f\n", ts.b[j]);
        // printf("c = %f\n", ts.c[j]);
        // printf("d = %f\n", ts.d[j]);
    }

    double* y = calloc(s->N + 1, sizeof(*y));
    run_thomas(&ts, &y[1]);
    y[0] = 0;
    y[s->N] = 0;

    return create_solution_t(s->x, y, s->N);
}

#define M_PI 3.14159265358979323846

int main(int argc, char** argv) {
    double lambda = 1;
    int n = 100;
    double l = M_PI / sqrt(lambda);

    fem_solver_t* fs = create_fem_solver_t(lambda, l, n);

    solution_t* s = run_fem(fs);

    int nn = 10;
    double nh = l / nn;
    for (int i = 0; i < nn; i++) {
        double x = i * nh;
        double my_val = function(s, x);
        double real_val = sin(sqrt(lambda) * x);
        printf("[x=%f]: real: %f | my: %f\n", x, real_val, my_val);
    }
}