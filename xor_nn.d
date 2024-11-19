module xor_nn;
import std;
import core.stdc.stdlib;
import std.random;
import std.datetime.systime : Clock;

struct Mat {
    int rows;
    int cols;
    int stride;
    float* es;
}

struct Xor {
    Mat a0;
    Mat w1, b1, a1;
    Mat w2, b2, a2;
}

float sigmoidf(float x) {
    // exp is the base e exponential function in D
    return 1.0f / (1.0f + exp(-x));
}

float rand_float(float low, float high, ref Random rnd)
{
    return uniform(low, high, rnd);
}

ref float MAT_AT(Mat m, int i, int j) {
    return (*(m.es + i*m.stride + j));
}

Xor xor_alloc() {
    Xor m;
    m.a0 = mat_alloc(1, 2);

    m.w1 = mat_alloc(2, 2);
    m.b1 = mat_alloc(1, 2);
    m.a1 = mat_alloc(1, 2);

    m.w2 = mat_alloc(2, 1);
    m.b2 = mat_alloc(1, 1);
    m.a2 = mat_alloc(1, 1);
    return m;
}

Mat mat_row(Mat m, int row) {
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

void mat_print(Mat m, string name) {
    writefln("%s = [", name);
    for(int i = 0; i < m.rows; i++) {
        string row = "  ";
        for(int j = 0; j < m.cols; j++) { 
            row ~= format("%.6f", MAT_AT(m, i, j));
            row ~= "    ";
        }
        writefln("%s", row);
    }
    writefln("]");
}

Mat mat_alloc(int rows, int cols) {
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

float cost(Xor m, Mat ti, Mat to) {
    assert(ti.rows == to.rows);
    assert(to.cols == m.a2.cols);
    int n = ti.rows;

    float c = 0.0f;
    for(int i = 0; i < n; i++) {
        Mat x = mat_row(ti, i);
        Mat y = mat_row(to, i);
        mat_copy(m.a0, x);
        forward_xor(m);

        int q = to.cols;
        for (int j = 0; j < q; j++) {
            float d = (MAT_AT(m.a2, 0, j) - MAT_AT(y, 0, j));
            c += d*d;
        }
    }
    return c/n;
}

float forward_xor(Xor m) {
    mat_dot(m.a1, m.a0, m.w1);
    mat_sum(m.a1, m.b1);
    mat_sig(m.a1);

    mat_dot(m.a2, m.a1, m.w2);
    mat_sum(m.a2, m.b2);
    mat_sig(m.a2);

    return *(m.a2.es);
}

void finite_diff(Xor m, Xor g, float eps, Mat ti, Mat to)
{
    float saved;

    // First neuron weight w1 and b1 are not being taken into account
    float c = cost(m, ti, to);
    for (int i = 0; i < m.w1.rows; i++) {
        for (int j = 0; j < m.w1.cols; j++) {   
            saved = MAT_AT(m.w1, i, j);
            MAT_AT(m.w1, i, j) += eps;
            MAT_AT(g.w1, i, j) = (cost(m, ti, to) - c)/eps;
            MAT_AT(m.w1, i, j) = saved;
        }
    }

    for (int i = 0; i < m.b1.rows; i++) {
        for (int j = 0; j < m.b1.cols; j++) {   
            saved = MAT_AT(m.b1, i, j);
            MAT_AT(m.b1, i, j) += eps;
            MAT_AT(g.b1, i, j) = (cost(m, ti, to) - c)/eps;
            MAT_AT(m.b1, i, j) = saved;
        }
    }

    for (int i = 0; i < m.w2.rows; i++) {
        for (int j = 0; j < m.w2.cols; j++) {   
            saved = MAT_AT(m.w2, i, j);
            MAT_AT(m.w2, i, j) += eps;
            MAT_AT(g.w2, i, j) = (cost(m, ti, to) - c)/eps;
            MAT_AT(m.w2, i, j) = saved;
        }
    }

    for (int i = 0; i < m.b2.rows; i++) {
        for (int j = 0; j < m.b2.cols; j++) {   
            saved = MAT_AT(m.b2, i, j);
            MAT_AT(m.b2, i, j) += eps;
            MAT_AT(g.b2, i, j) = (cost(m, ti, to) - c)/eps;
            MAT_AT(m.b2, i, j) = saved;
        }
    }

    //mat_print(g.w1, "g.w1");
    //mat_print(g.b1, "g.b1");
    //mat_print(g.w2, "g.w2");
    //mat_print(g.b2, "g.b2");
}

void xor_learn(Xor m, Xor g, float rate)
{
    for (int i = 0; i < m.w1.rows; i++) {
        for (int j = 0; j < m.w1.cols; j++) {   
            MAT_AT(m.w1, i, j) -= rate*MAT_AT(g.w1, i, j);
        }
    }

    for (int i = 0; i < m.b1.rows; i++) {
        for (int j = 0; j < m.b1.cols; j++) {   
            MAT_AT(m.b1, i, j) -= rate*MAT_AT(g.b1, i, j);
        }
    }

    for (int i = 0; i < m.w2.rows; i++) {
        for (int j = 0; j < m.w2.cols; j++) {   
            MAT_AT(m.w2, i, j) -= rate*MAT_AT(g.w2, i, j);
        }
    }

    for (int i = 0; i < m.b2.rows; i++) {
        for (int j = 0; j < m.b2.cols; j++) {   
            MAT_AT(m.b2, i, j) -= rate*MAT_AT(g.b2, i, j);
        }
    }
}

void xor() {
    auto rnd = Random(cast(int) Clock.currTime().toUnixTime());
    float[4] id_data = [1, 0, 0, 1];
    
    Xor m = xor_alloc();
    Xor g = xor_alloc();
    
    mat_rand(m.w1, rnd, 0, 1);
    mat_rand(m.b1, rnd, 0, 1);
    mat_rand(m.w2, rnd, 0, 1);
    mat_rand(m.b2, rnd, 0, 1);

    float[] td = [
        0, 0, 0,
        1, 0, 1,
        0, 1, 1,
        1, 1, 0
    ];

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

    float eps = 1e-1;
    float rate = 1e-1;

    for(int i = 0; i < 100000; i++) {
        finite_diff(m, g, eps, ti, to);
        xor_learn(m, g, rate);
        writefln("cost = %s", cost(m, ti, to));
    }

    for(int i = 0; i < 2; i++) {
        for(int j = 0; j < 2; j++) {
            MAT_AT(m.a0, 0, 0) = i;
            MAT_AT(m.a0, 0, 1) = j;
            forward_xor(m);
            float y = *m.a2.es;

            writefln("%s ^ %s = %s", i, j, y);
        }
    }

}