module nn;
import std;
import core.stdc.stdlib;
import std.random;
import std.datetime.systime : Clock;

struct Mat {
    ulong rows;
    ulong cols;
    ulong stride;
    float* es;
}

struct NN {
    size_t count;
    Mat* bs;
    Mat* ws;
    Mat* as;
}

float sigmoidf(float x) {
    // exp is the base e exponential function in D
    return 1.0f / (1.0f + exp(-x));
}

float rand_float(float low, float high, ref Random rnd)
{
    return uniform(low, high, rnd);
}

ref float MAT_AT(Mat m, size_t i, size_t j) {
    return (*(m.es + i*m.stride + j));
}

Mat NN_OUTPUT(NN nn) {
    return nn.as[nn.count];
}

void nn_print(NN nn) {
    
    writefln("nn = [");

    mat_print(nn.as[0], "a0", 4);
    for(int i = 0; i < nn.count; i++) {
        mat_print(nn.ws[i], format("w%s", i+1), 4);
        mat_print(nn.bs[i], format("b%s", i+1), 4);
        mat_print(nn.as[i + 1], format("a%s", i+1), 4);
    }
    writefln("]");
}

void nn_rand(NN nn, ref Random rnd, float low, float high) {

    for(int i = 0; i < nn.count; i++) {
        mat_rand(nn.ws[i], rnd, low, high);
        mat_rand(nn.bs[i], rnd, low, high);
        mat_rand(nn.as[i + 1], rnd, low, high);
    }
}

NN nn_alloc(ulong* arch, ulong arch_count) {
    NN nn;

    assert(arch[0] > 0, "Inputs should have non-zero columns");
    nn.count = arch_count - 1;

    nn.ws = cast(Mat*) malloc(nn.count*(*nn.ws).sizeof);
    assert(nn.ws != null, "Null nn weight matrixes");
    nn.bs = cast(Mat*) malloc(nn.count*(*nn.bs).sizeof);
    assert(nn.bs != null, "Null nn bias matrixes");
    nn.as = cast(Mat*) malloc((nn.count + 1)*(*nn.as).sizeof);
    assert(nn.as != null, "Null nn activation matrixes");

    nn.as[0] = mat_alloc(1, arch[0]);

    for(int i = 0; i < nn.count; i++) {
        nn.ws[i] = mat_alloc(nn.as[i].cols, arch[i + 1]);
        nn.bs[i] = mat_alloc(1, arch[i + 1]);
        nn.as[i + 1] = mat_alloc(1, arch[i + 1]);
        mat_fill(nn.ws[i], 0);
        mat_fill(nn.bs[i], 1);
        mat_fill(nn.as[i + 1], 2);
    }
    return nn;
}

Mat mat_row(Mat m, size_t row) {
    assert(row < m.rows, "Trying to access rows outside the range of the matrix");
    float* start = &MAT_AT(m, row, 0);
    Mat s = {
        1,
        m.cols,
        m.stride,
        start
    };
    return s;
}

void mat_copy(Mat dst, Mat src) {
    assert(src.rows == dst.rows);
    assert(src.cols == dst.cols);

    for(int i = 0; i < src.rows; i++) {
        for(int j = 0; j < src.cols; j++) {
            MAT_AT(dst, i, j) = MAT_AT(src, i, j);
        }
    }
}

void mat_dot(Mat dst, Mat a, Mat b) {
    assert(a.cols == b.rows, "Matrixes not multiplicable, cols of a not equal to rows of b");
    auto n = a.cols;
    
    assert(a.rows == dst.rows, "Invalid dst matrix, rows of dst not equal to rows of a");
    assert(b.cols == dst.cols, "Invalid dst matrix, cols of dst not equal to cols of b");


    for(int i = 0; i < dst.rows; i++) {
        for(int j = 0; j < dst.cols; j++) {
            MAT_AT(dst, i, j) = 0;
            for(int k = 0; k < n; k++) {
                MAT_AT(dst, i, j) += MAT_AT(a, i, k) * MAT_AT(b, k, j); 
            } 
        }
    }
}

void mat_sum(Mat dst, Mat a) {
    assert(dst.rows == a.rows, "Matrixes not summable, unequal rows in source and dst");
    assert(dst.cols == a.cols, "Matrixes not summable, unequal cols in source and dst");

    for(int i = 0; i < dst.rows; i++) {
        for(int j = 0; j < dst.cols; j++) { 
            MAT_AT(dst, i, j) += MAT_AT(a, i, j); 
        }
    }
}

void mat_print(Mat m, string name, ulong padding) {
    string pd = "";
    for(int i = 0; i < padding; i++) {
        pd ~= " ";
    }
    writefln("%s%s = [", pd, name);
    
    for(int i = 0; i < m.rows; i++) {
        string row = "  ";
        for(int j = 0; j < m.cols; j++) { 
            row ~= format("%.6f", MAT_AT(m, i, j));
            row ~= "    ";
        }
        writefln("%s%s", pd, row);
    }
    writefln("%s]", pd);
}

Mat mat_alloc(ulong rows, ulong cols) {
    Mat m;
    m.rows = rows;
    m.cols = cols;
    m.stride = cols;
    m.es = cast(float*) malloc((*m.es).sizeof*rows*cols);
    assert(m.es != null, "Allocation is null");
    return m;
}

void mat_rand(Mat m, ref Random rnd, float low, float high) {
    for(int i = 0; i < m.rows; i++) {
        for(int j = 0; j < m.cols; j++) {
            MAT_AT(m, i, j) = rand_float(low, high, rnd);
        }
    }
}

void mat_fill(Mat m, float fill) {
    for(int i = 0; i < m.rows; i++) {
        for(int j = 0; j < m.cols; j++) {
            MAT_AT(m, i, j) = fill;
        }
    }
}

void mat_sig(Mat m) {
    for(int i = 0; i < m.rows; i++) {
        for(int j = 0; j < m.cols; j++) {
            MAT_AT(m, i, j) = sigmoidf(MAT_AT(m, i, j));
        }
    }
}

float cost_nn(NN nn, Mat ti, Mat to) {

    assert(ti.rows == to.rows);
    assert(to.cols == NN_OUTPUT(nn).cols);
    ulong n = ti.rows;

    float c = 0.0f;
    for(size_t i = 0; i < n; i++) {
        Mat x = mat_row(ti, i);
        Mat y = mat_row(to, i);
        mat_copy(nn.as[0], x);
        forward_nn(nn);

        ulong q = to.cols;
        for (int j = 0; j < q; j++) {
            float d = (MAT_AT(NN_OUTPUT(nn), 0, j) - MAT_AT(y, 0, j));
            c += d*d;
        }
    }
    return c/n;
}

void nn_zero(NN nn) {
    for(int i = 0; i < nn.count; i++) {
        mat_fill(nn.as[i], 0);
        mat_fill(nn.ws[i], 0);
        mat_fill(nn.bs[i], 0);
    }
    mat_fill(nn.as[nn.count], 0);
}

void forward_nn(NN nn) {
    for(int i = 0; i < nn.count; i++) {
        mat_dot(nn.as[i + 1], nn.as[i], nn.ws[i]);
        mat_sum(nn.as[i + 1], nn.bs[i]);
        mat_sig(nn.as[i + 1]);
    }
}

void finite_diff_nn(NN nn, NN g, float eps, Mat ti, Mat to)
{    
    float c = cost_nn(nn, ti, to);
    float saved;
    for(int i=0; i < nn.count; i++) {
        for(int j=0; j < nn.ws[i].rows; j++) {
            for(int k=0; k < nn.ws[i].cols; k++) {
                saved = MAT_AT(nn.ws[i], j, k);
                MAT_AT(nn.ws[i], j, k) += eps;
                MAT_AT(g.ws[i], j, k) = (cost_nn(nn, ti, to) - c)/eps;
                MAT_AT(nn.ws[i], j, k) = saved;           
            }
        }
        for(int j=0; j < nn.bs[i].rows; j++) {
            for(int k=0; k < nn.bs[i].cols; k++) {
                saved = MAT_AT(nn.bs[i], j, k);
                MAT_AT(nn.bs[i], j, k) += eps;
                MAT_AT(g.bs[i], j, k) = (cost_nn(nn, ti, to) - c)/eps;
                MAT_AT(nn.bs[i], j, k) = saved;           
            }
        }
    }
}

void nn_backprop(NN nn, NN g, Mat ti, Mat to)
{
    assert(ti.rows == to.rows);
    size_t n = ti.rows;
    assert(NN_OUTPUT(nn).cols == to.cols);

    nn_zero(g);
    for(size_t i = 0; i < n; i++) {
        mat_copy(nn.as[0], mat_row(ti, i));
        forward_nn(nn);
        
        for(size_t j = 0; j <= nn.count; j++) {
            mat_fill(g.as[j], 0);
        }

        for(size_t j = 0; j < NN_OUTPUT(nn).cols; j++) {
            MAT_AT(NN_OUTPUT(g), 0, j) = MAT_AT(NN_OUTPUT(nn), 0, j) - MAT_AT(to, i, j);
        }

        for(size_t l = nn.count; l > 0; l--) {
            for(size_t j = 0; j < nn.as[l].cols; j++) {
                float a = MAT_AT(nn.as[l], 0, j);
                float da = MAT_AT(g.as[l], 0, j);
                MAT_AT(g.bs[l - 1], 0, j) += 2*da*a*(1-a);
                for(size_t k = 0; k < nn.as[l - 1].cols; k++) {
                    float pa = MAT_AT(nn.as[l-1], 0, k);
                    float w = MAT_AT(nn.ws[l-1], k, j);
                    MAT_AT(g.ws[l - 1], k, j) += 2*da*a*(1-a)*pa;
                    MAT_AT(g.as[l - 1], k, j) += 2*da*a*(1-a)*w;
                }
            }
        }
    }

    for(size_t i = 0; i < g.count; i++) {
        for(size_t j = 0; j < g.ws[i].rows; j++) {
            for(size_t k = 0; k < g.ws[i].cols; k++) {
                MAT_AT(g.ws[i], j, k) /= n;
            }
        }
        for(size_t j = 0; j < g.bs[i].rows; j++) {
            for(size_t k = 0; k < g.bs[i].cols; k++) {
                MAT_AT(g.bs[i], j, k) /= n;
            }
        }
    }
}

void learn_nn(NN nn, NN g, float rate)
{
    for (int i = 0; i < nn.count; i++) {
        for (int j = 0; j < nn.ws[i].rows; j++) {
            for (int k = 0; k < nn.ws[i].cols; k++) {   
                MAT_AT(nn.ws[i], j, k) -= rate*MAT_AT(g.ws[i], j, k);
            }
        }
        for (int j = 0; j < nn.bs[i].rows; j++) {
            for (int k = 0; k < nn.bs[i].cols; k++) {   
                MAT_AT(nn.bs[i], j, k) -= rate*MAT_AT(g.bs[i], j, k);
            }
        }
    }
}

void test() {
    auto rnd = Random(cast(int) Clock.currTime().toUnixTime());
    float[4] id_data = [1, 0, 0, 1];

    float[] td_xor = [
        0, 0, 0,
        1, 0, 1,
        0, 1, 1,
        1, 1, 0
    ];

    float[] td_or = [
        0, 0, 0,
        1, 0, 1,
        0, 1, 1,
        1, 1, 1
    ];

    float[] td = td_xor;

    int stride = 3;
    int n = cast(int) td.length/stride;

    Mat ti = {
        n,
        2,
        stride,
        td.ptr
    };

    Mat to = {
        n,
        1,
        stride,
        td.ptr + 2
    };

    float eps = 1e-3;
    float rate = 1;
    
    ulong[] arch = [2, 2, 1];
    NN nn = nn_alloc(arch.ptr, arch.length);
    NN g  = nn_alloc(arch.ptr, arch.length);
    nn_rand(nn, rnd, 0, 1);

    writefln("cost: %s",cost_nn(nn, ti, to));
    for(int i = 0; i < 20000; i++) {
        //finite_diff_nn(nn, g, eps, ti, to);
        nn_backprop(nn, g, ti, to);
        learn_nn(nn, g, rate);
        writefln("cost: %s", cost_nn(nn, ti, to));
    }

    for(ulong i = 0; i < 2; i++) {
        for(ulong j = 0; j < 2; j++) {
            MAT_AT(nn.as[0], 0, 0) = i;
            MAT_AT(nn.as[0], 0, 1) = j;
            forward_nn(nn);
            writefln("%s ^ %s = %s", i, j, MAT_AT(nn.as[arch.length - 1], 0, 0));
        }
    }
}

void adder() {
    auto rnd = Random(cast(int) Clock.currTime().toUnixTime());
    static size_t BITS = 2;

    size_t n = 1<<BITS;
    size_t rows = n*n;
    Mat ti = mat_alloc(rows, 2*BITS);
    Mat to = mat_alloc(rows, BITS+1);
    for(size_t i = 0; i < ti.rows; i++) {
        size_t x = i/n;
        size_t y = i%n;
        size_t z = x + y;    
        for(size_t j = 0; j < BITS; j++) {
            MAT_AT(ti, i, j)      = (x>>j)&1;
            MAT_AT(ti, i, j+BITS) = (y>>j)&1;
            MAT_AT(to, i, j)      = (z>>j)&1;
        }
        MAT_AT(to, i, BITS) = z >= n;    
    }
    //mat_print(ti, "ti", 10);
    //mat_print(to, "to", 10);

    ulong[] arch = [2*BITS, BITS,BITS+1];
    NN nn = nn_alloc(arch.ptr, arch.length);
    NN g  = nn_alloc(arch.ptr, arch.length);
    nn_rand(nn, rnd, 0, 1);
    nn_print(nn);

    float rate = 1;

    writefln("cost = %s", cost_nn(nn, ti, to));
    nn_print(nn);
    for(size_t i = 0; i < 50*10000; i++) {
        nn_backprop(nn, g, ti, to);
        learn_nn(nn, g, rate);
        writefln("%s: cost = %s", i, cost_nn(nn, ti, to));
    }
    //nn_print(g);
}