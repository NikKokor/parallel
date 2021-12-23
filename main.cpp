#include <thread>
#include <mutex>
#include <vector>
#include <iostream>
#include <omp.h>
#include <atomic>
#include <cstdlib>
#include <fstream>
#include <algorithm>
#include <cstring>
#include <type_traits>
#include "barrier.h"

#define STEPS 100000000
#define CACHE_LINE 64u
#define A -1
#define B 1

using namespace std;

typedef double (*f_t)(double);

typedef double (*E_t)(double, double, f_t);

struct ExperimentResult {
    double result;
    double time;
};
struct partialSumT {
    alignas(64) double val;
};

#if defined(__GNUC__)
namespace std {
    constexpr std::size_t hardware_constructive_interference_size = 64u;
    constexpr std::size_t hardware_destructive_interference_size = 64u;
}
#endif

size_t ceil_div(size_t x, size_t y) {
    return (x + y - 1) / y;
}

unsigned g_num_threads = thread::hardware_concurrency();

void set_num_threads(unsigned T) {
    g_num_threads = T;
    omp_set_num_threads(T);
}

unsigned get_num_threads() {
    return g_num_threads;
}

typedef double (*I_t)(double, double, f_t);

using namespace std;

double f(double x) {
    return x * x;
}

double integrateDefault(double a, double b, f_t f) {
    double result = 0, dx = (b - a) / STEPS;

    for (unsigned i = 0; i < STEPS; i++) {
        result += f(i * dx + a);
    }

    return result * dx;
}

double integrateCrit(double a, double b, f_t f) {
    double result = 0, dx = (b - a) / STEPS;

#pragma omp parallel
    {
        double R = 0;
        unsigned t = (unsigned) omp_get_thread_num();
        unsigned T = (unsigned) omp_get_num_threads();

        for (unsigned i = t; i < STEPS; i += T) {
            R += f(i * dx + a);
        }
#pragma omp critical
        result += R;
    }
    return result * dx;
}

double integrateMutex(double a, double b, f_t f) {
    unsigned T = thread::hardware_concurrency();
    mutex mtx;
    vector<thread> threads;
    double result = 0, dx = (b - a) / STEPS;

    for (unsigned t = 0; t < T; t++) {
        threads.emplace_back([=, &result, &mtx]() {
            double R = 0;
            for (unsigned i = t; i < STEPS; i += T) {
                R += f(i * dx + a);
            }

            {
                scoped_lock lck{mtx};
                result += R;
            }
        });

    }
    for (auto &thr: threads) {
        thr.join();
    }

    return result * dx;
}

double integrateArr(double a, double b, f_t f) {
    unsigned T;
    double result = 0, dx = (b - a) / STEPS;
    double *accum;

#pragma omp parallel shared(accum, T)
    {
        unsigned t = (unsigned) omp_get_thread_num();
#pragma omp single
        {
            T = (unsigned) omp_get_num_threads();
            accum = (double *) calloc(T, sizeof(double));
            //accum1.reserve(T);
        }

        for (unsigned i = t; i < STEPS; i += T) {
            accum[t] += f(dx * i + a);
        }
    }

    for (unsigned i = 0; i < T; ++i) {
        result += accum[i];
    }

    free(accum);

    return result * dx;
}

double integrateArrAlign(double a, double b, f_t f) {
    unsigned int T;
    double result = 0, dx = (b - a) / STEPS;
    partialSumT *accum = 0;

#pragma omp parallel shared(accum, T)
    {
        unsigned int t = (unsigned int) omp_get_thread_num();
#pragma omp single
        {
            T = (unsigned int) omp_get_num_threads();
            accum = (partialSumT *) aligned_alloc(CACHE_LINE, T * sizeof(partialSumT));
        }

        for (unsigned int i = t; i < STEPS; i += T) {
            accum[t].val += f(dx * i + a);
        }
    }

    for (unsigned int i = 0; i < T; i++) {
        result += accum[i].val;
    }

    free(accum);

    return result * dx;
}

double integrateReductionOMP(double a, double b, f_t f) {
    double result = 0, dx = (b - a) / STEPS;

#pragma omp parallel for reduction(+: result)
    for (unsigned int i = 0; i < STEPS; ++i) {
        result += f(dx * i + a);
    }

    return result * dx;
}

double integratePS(double a, double b, f_t f) {
    double dx = (b - a) / STEPS;
    double result = 0;
    unsigned T = get_num_threads();
    auto vec = vector(T, partialSumT{0.0});
    vector<thread> threadVec;

    auto threadProc = [=, &vec](auto t) {
        for (auto i = t; i < STEPS; i += T) {
            vec[t].val += f(dx * i + a);
        }
    };

    for (auto t = 1; t < T; t++) {
        threadVec.emplace_back(threadProc, t);
    }

    threadProc(0);

    for (auto &thread: threadVec) {
        thread.join();
    }

    for (auto elem: vec) {
        result += elem.val;
    }

    return result * dx;
}

double integrateAtomic(double a, double b, f_t f) {
    vector<thread> threads;
    std::atomic<double> result = {0};
    double dx = (b - a) / STEPS;
    unsigned int T = get_num_threads();

    auto fun = [dx, &result, f, a, T](auto t) {
        double R = 0;
        for (unsigned i = t; i < STEPS; i += T) {
            R += f(i * dx + a);
        }

        result = result + R;
    };

    for (unsigned int t = 1; t < T; ++t) {
        threads.emplace_back(fun, t);
    }

    fun(0);

    for (auto &thr: threads) {
        thr.join();
    }

    return result * dx;
}

template <class ElementType, class BinaryFn>
ElementType reduce_vector(const ElementType* V, size_t n, BinaryFn f, ElementType zero)
{
    unsigned T = get_num_threads();
    struct reduction_partial_result_t
    {
        alignas(hardware_destructive_interference_size) ElementType value;
    };
    static auto reduction_partial_results =
            vector<reduction_partial_result_t>(thread::hardware_concurrency(),
                                               reduction_partial_result_t{zero});
    constexpr size_t k = 2;
    barrier bar {T};

    auto thread_proc = [=, &bar](unsigned t)
    {
        auto K = ceil_div(n, k);
        size_t Mt = K / T;
        size_t it1 = K % T;

        if(t < it1)
        {
            it1 = ++Mt * t;
        }
        else
        {
            it1 = Mt * t + it1;
        }
        it1 *= k;
        size_t mt = Mt * k;
        size_t it2 = it1 + mt;

        ElementType accum = zero;
        for(size_t i = it1; i < it2; i++)
            accum = f(accum, V[i]);

        reduction_partial_results[t].value = accum;

#if 0
        size_t s = 1;
        while(s < T)
        {
            bar.arrive_and_wait();
            if((t % (s * k)) && (t + s < T))
            {
                reduction_partial_results[t].value = f(reduction_partial_results[t].value,
                                                       reduction_partial_results[t + s].value);
                s *= k;
            }
        }
#else
        for(std::size_t s = 1, s_next = 2; s < T; s = s_next, s_next += s_next)
        {
            bar.arrive_and_wait();
            if(((t % s_next) == 0) && (t + s < T))
                reduction_partial_results[t].value = f(reduction_partial_results[t].value,
                                                       reduction_partial_results[t + s].value);
        }
#endif
    };

    vector<thread> threads;
    for(unsigned t = 1; t < T; t++)
        threads.emplace_back(thread_proc, t);
    thread_proc(0);
    for(auto& thread : threads)
        thread.join();

    return reduction_partial_results[0].value;
}

template <class ElementType, class UnaryFn, class BinaryFn>
#if 0
requires {
    is_invocable_r_v<UnaryFn, ElementType, ElementType> &&
    is_invocable_r_v<BinaryFn, ElementType, ElementType, ElementType>
}
#endif
ElementType reduce_range(ElementType a, ElementType b, size_t n, UnaryFn get, BinaryFn reduce_2, ElementType zero)
{
    unsigned T = get_num_threads();
    struct reduction_partial_result_t
    {
        alignas(hardware_destructive_interference_size) ElementType value;
    };
    static auto reduction_partial_results =
            vector<reduction_partial_result_t>(thread::hardware_concurrency(), reduction_partial_result_t{zero});

    barrier bar{T};
    constexpr size_t k = 2;
    auto thread_proc = [=, &bar](unsigned t)
    {
        auto K = ceil_div(n, k);
        double dx = (b - a) / n;
        size_t Mt = K / T;
        size_t it1 = K % T;

        if(t < it1)
        {
            it1 = ++Mt * t;
        }
        else
        {
            it1 = Mt * t + it1;
        }
        it1 *= k;
        size_t mt = Mt * k;
        size_t it2 = it1 + mt;

        ElementType accum = zero;
        for(size_t i = it1; i < it2; i++)
            accum = reduce_2(accum, get(a + i*dx));

        reduction_partial_results[t].value = accum;

        for(size_t s = 1, s_next = 2; s < T; s = s_next, s_next += s_next)
        {
            bar.arrive_and_wait();
            if(((t % s_next) == 0) && (t + s < T))
                reduction_partial_results[t].value = reduce_2(reduction_partial_results[t].value,
                                                              reduction_partial_results[t + s].value);
        }
    };

    vector<thread> threads;
    for(unsigned t = 1; t < T; t++)
        threads.emplace_back(thread_proc, t);
    thread_proc(0);
    for(auto& thread : threads)
        thread.join();
    return reduction_partial_results[0].value;
}

double integrateReduction(double a, double b, f_t f){
    return reduce_range(a, b, STEPS, f, [](auto x, auto y) {return x + y;}, 0.0) * ((b - a) / STEPS);
}

ExperimentResult runExperiment(I_t I) {
    double t0, t1, result;

    t0 = omp_get_wtime();
    result = I(A, B, f);
    t1 = omp_get_wtime();

    return {result, t1 - t0};
}

void showExperimentResults(I_t I, ofstream &file) {
    set_num_threads(1);
    ExperimentResult R = runExperiment(I);
    double T1 = R.time;

    file.width(10);
    file << "Threads,";
    file.width(10);
    file << "Time,";
    file.width(14);
    file << "Acceleration,";
    file.width(10);
    file << "Result" << endl;
    printf("%10s,%10s,%14s,%10s\n", "Threads", "Time", "Acceleration", "Result");

    file.width(10);
    file << "1,";
    file.width(10);
    file.precision(6);
    file << to_string(R.time) + ",";
    file.width(14);
    file.precision(6);
    file << to_string(T1 / R.time) + ",";
    file.width(10);
    file.precision(6);
    file << to_string(R.result) << endl;
    printf("%10d,%10g,%14g,%10g\n", 1, R.time, T1 / R.time, R.result);

    for (int T = 2; T <= omp_get_num_procs(); ++T) {
        set_num_threads(T);
        ExperimentResult result = runExperiment(I);
        file.width(9);
        file << T << ",";
        file.width(10);
        file.precision(6);
        file << to_string(result.time) + ",";
        file.width(14);
        file.precision(6);
        file << to_string(T1 / result.time) + ",";
        file.width(10);
        file.precision(6);
        file << to_string(result.result) << endl;
        // printf("%10g\t%10g\t%10g\n", result.result, result.time, T1/result.time);
        printf("%10d,%10g,%14g,%10g\n", T, result.time, T1 / result.time, result.result);
    }

    file << endl;
    cout << endl;
}

int main() {
    ifstream _file;
    ofstream file;
    int i = 1;
    for (i; i < 20; i++) {
        string str = "results" + to_string(i) + ".txt";
        _file.open(str);
        if (!_file.good()) {
            _file.close();
            cout << "file create\n";
            file.open(str);
            i = 20;
        } else
            _file.close();
    }
    file << "integrateArrAlign" << endl;
    showExperimentResults(integrateArrAlign, file);
    //file << "integrateDefault" << endl;
    //showExperimentResults(integrateDefault, file);
    file << "integrateCrit" << endl;
    showExperimentResults(integrateCrit, file);
    //file << "integrateMutex" << endl;
    //showExperimentResults(integrateMutex, file);
    //file << "integrateArr" << endl;
    //showExperimentResults(integrateArr, file);
    file << "integrateReductionOMP" << endl;
    showExperimentResults(integrateReductionOMP, file);
    file << "integratePS" << endl;
    showExperimentResults(integratePS, file);
    //file << "integrateAtomic" << endl;
    //showExperimentResults(integrateAtomic, file);
    file << "integrateReduction" << endl;
    showExperimentResults(integrateReduction, file);

    return 0;
}
