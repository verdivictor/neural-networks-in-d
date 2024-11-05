import std.stdio;
import std.algorithm;
import std.range;
import std.random;
import std.datetime.systime : Clock;
import std.math;

// XOR modelling

float rand_float(int b)
{
    auto utcDateTime = Clock.currTime();
    auto rnd = Random(cast(int) utcDateTime.toUnixTime() + b);
    //auto rnd = Random(10);
    auto r = uniform(0.0L, 100.0L, rnd) / 100.0L;
    return r * 1.0L;
}

struct Xor {
    float or_w1;
    float or_w2;
    float or_b;
    float nand_w1;
    float nand_w2;
    float nand_b;
    float and_w1;
    float and_w2;
    float and_b;
}

float[3][] train = [
    [0, 0, 0],
    [0, 1, 1],
    [1, 0, 1],
    [1, 1, 0],
];

float cost(Xor m) {
    ulong train_len = train.length;
    float result = 0.0f;
    for(int i = 0; i < train_len; i++) {
        float x1 = train[i][0];
        float x2 = train[i][1];
        float y = forward(m, x1, x2);
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

float forward(Xor m, float x1, float x2) {
    float a = sigmoid(m.or_w1*x1 + m.or_w2*x2 + m.or_b);
    float b = sigmoid(m.nand_w1*x1 + m.nand_w2*x2 + m.nand_b);
    return sigmoid(a*m.and_w1 + b*m.and_w2 + m.and_b);
}

Xor finite_diff(Xor m) {
    Xor g;
    float c = cost(m);
    float saved;
    float eps = 1e-1;

    saved = m.or_w1;
    m.or_w1 += eps;
    g.or_w1 = (cost(m) - c)/eps;
    m.or_w1 = saved;

    saved = m.or_w2;
    m.or_w2 += eps;
    g.or_w2 = (cost(m) - c)/eps;
    m.or_w2 = saved;
    
    saved = m.or_b;
    m.or_b += eps;
    g.or_b = (cost(m) - c)/eps;
    m.or_b = saved;

    saved = m.nand_w1;
    m.nand_w1 += eps;
    g.nand_w1 = (cost(m) - c)/eps;
    m.nand_w1 = saved;

    saved = m.nand_w2;
    m.nand_w2 += eps;
    g.nand_w2 = (cost(m) - c)/eps;
    m.nand_w2 = saved;
    
    saved = m.nand_b;
    m.nand_b += eps;
    g.nand_b = (cost(m) - c)/eps;
    m.nand_b = saved;

    saved = m.and_w1;
    m.and_w1 += eps;
    g.and_w1 = (cost(m) - c)/eps;
    m.and_w1 = saved;
    
    saved = m.and_w2;
    m.and_w2 += eps;
    g.and_w2 = (cost(m) - c)/eps;
    m.and_w2 = saved;
    
    saved = m.and_b;
    m.and_b += eps;
    g.and_b = (cost(m) - c)/eps;
    m.and_b = saved;

    return g;
}

Xor rand_xor() {
    Xor m;
    m.or_w1 = rand_float(2);
    m.or_w2 = rand_float(3);
    m.or_b = rand_float(4);
    m.nand_w1 = rand_float(5);
    m.nand_w2 = rand_float(6);
    m.nand_b = rand_float(7);
    m.and_w1 = rand_float(8);
    m.and_w2 = rand_float(9);
    m.and_b = rand_float(10);
    return m;
}

Xor learn(Xor m, Xor g, float rate) {
    m.or_w1 -= g.or_w1*rate;
    m.or_w2 -= g.or_w2*rate;
    m.or_b -= g.or_b*rate;
    m.nand_w1 -= g.nand_w1*rate;
    m.nand_w2 -= g.nand_w2*rate;
    m.nand_b -= g.nand_b*rate;
    m.and_w1 -= g.and_w1*rate;
    m.and_w2 -= g.and_w2*rate;
    m.and_b -= g.and_b*rate;
    return m;
}

void print_xor(Xor m) {
    writefln("%s", m.or_w1);
    writefln("%s", m.or_w2);
    writefln("%s", m.or_b);
    writefln("%s", m.nand_w1);
    writefln("%s", m.nand_w2);
    writefln("%s", m.nand_b);
    writefln("%s", m.and_w1);
    writefln("%s", m.and_w2);
    writefln("%s", m.and_b);
}

int main() 
{
    Xor m = rand_xor();
    float rate = 1;
    print_xor(m);
    writefln("cost = %s", cost(m));
    
    for(int i = 0; i < 100000; i++) {
        Xor g = finite_diff(m);
        //print_xor(g);
        m = learn(m, g, rate);
        //print_xor(m);
    }

    // Sometimes the algorithm will find a path through OR, AND, etc. 
    // Points to be crossed are supplied, but the "network" will find
    // its own path. Sometimes first step is AND, or OR'ish, or XOR'ish.    
    writefln("cost = %s", cost(m));

    writefln("XOR:");
    for(int i = 0; i < 2; i++) {
        for(int j = 0; j < 2; j++) {
            writefln("%s ^ %s = %s", i, j, forward(m, i, j));
        }
    }
    writefln("--------------");

    writefln("OR Neuron:");
    for(int i = 0; i < 2; i++) {
        for(int j = 0; j < 2; j++) {
            writefln("%s | %s = %s", i, j, sigmoid(m.or_w1*i + m.or_w2*j + m.or_b));
        }
    }
    writefln("--------------");

    writefln("NAND Neuron:");
    for(int i = 0; i < 2; i++) {
        for(int j = 0; j < 2; j++) {
            writefln("~(%s & %s) = %s", i, j, sigmoid(m.nand_w1*i + m.nand_w2*j + m.nand_b));
        }
    }
    writefln("--------------");

    writefln("AND Neuron:");
    for(int i = 0; i < 2; i++) {
        for(int j = 0; j < 2; j++) {
            writefln("%s & %s = %s", i, j, sigmoid(m.and_w1*i + m.and_w2*j + m.and_b));
        }
    }
    writefln("--------------");
    return 0;
}