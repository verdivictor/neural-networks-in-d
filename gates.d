import std.stdio;
import std.algorithm;
import std.range;
import std.random;
import std.datetime.systime : Clock;
import std.math;

// OR, AND & NAND modelable
// XOR not modelable by a single neuron
float[3][] train = [
    [0, 0, 0],
    [0, 1, 1],
    [1, 0, 1],
    [1, 1, 1],
];

float cost(float w1, float w2, float b) {
    ulong train_len = train.length;
    float result = 0.0f;
    for(int i = 0; i < train_len; i++) {
        float x1 = train[i][0];
        float x2 = train[i][1];
        float y = sigmoid(x1*w1 + x2*w2 + b);
        float diff = y - train[i][2];
        result += diff*diff;
    }
    result /= train_len;
    return result;
}

float sigmoid(float x) {
    // exp is the base e exponential function in D
    return 1.0f / (1.0f + exp(-x));
}

void main()
{

    auto utcDateTime = Clock.currTime();
    auto rnd = Random(cast(int) utcDateTime.toUnixTime());
    //auto rnd = Random(10);
    auto r1 = uniform(0.0L, 100.0L, rnd) / 100.0L;
    auto r2 = uniform(0.0L, 100.0L, rnd) / 100.0L;

    float w1 = r1;
    float w2 = r2;
    float b = r2;

    //writefln("c = %s", cost(w1, w2, b));

    float eps = 1e-1;
    float rate = 1e-1;
    //writefln("w1 = %s, w2 = %s, b = %s, cost = %s", w1, w2, b, cost(w1, w2, b));

    for (int i =0; i < 100000; i++) {

        float c = cost(w1, w2, b);
        //writefln("%s", c);
        float dw1 = (cost(w1 + eps, w2, b) - c) / eps;
        float dw2 = (cost(w1, w2 + eps, b) - c) / eps;
        float db = (cost(w1, w2, b + eps) - c) / eps;

        w1 -= rate*dw1;
        w2 -= rate*dw2;
        b  -= rate*db;
    }
    //writefln("w1 = %s, w2 = %s, b = %s, cost = %s", w1, w2, b, cost(w1, w2, b));

    for(int i = 0; i < 2; i++) {
        for(int j = 0; j < 2; j++) {
            writefln("%s | %f = %f", i, j, sigmoid(i*w1 + j*w2 + b));
        }
    }    
}