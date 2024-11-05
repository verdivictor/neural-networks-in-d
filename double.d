import std.stdio;
import std.algorithm;
import std.range;
import std.random;
import std.datetime.systime : Clock;

float[2][] train = [
    [0, 0],
    [2, 6],
    [3, 9],
    [4, 12]
];

float cost(float w, float b) {
    ulong train_len = train.length;
    float result = 0.0L;
    for(int i = 0; i < train_len; i++) {
        float x = train[i][0];
        float y = train[i][0] * w + b;
        float diff = y - train[i][1];
        result += diff*diff;
    }
    result /= train_len;
    return result;
}

void main()
{

    // Get random float seeded from clock time
    auto utcDateTime = Clock.currTime();
    auto rnd = Random(cast(int) utcDateTime.toUnixTime());
    //auto rnd = Random(10);
    auto r = uniform(0.0L, 100.0L, rnd) / 100.0L;

    // Weight that satisfies the function
    float w = r * 10.0L;
    float b = r * 5.0L;

    float eps = 1e-3;
    float rate = 1e-3;

    writefln("cost  = %s, w = %s, b = %s", cost(w, b), w ,b);
    for(int i = 0; i < 50000; i++) {
        float c = cost(w, b);
        float dw = (cost(w + eps, b) - c) / eps;
        float db = (cost(w, b + eps) - c) / eps;
        w -= rate*dw;
        b -= rate*db; 
    }
    writefln("cost  = %s, w = %s, b = %s", cost(w, b), w ,b);
}