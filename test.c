#include <stdio.h>
#include <stdlib.h>
#include <math.h>

double sigmoid(double x) {
    return 1.f / (1.f + expf(-x));
}

void main ()

{
    printf("%f", sigmoid(20.0));
    printf("swag");
}