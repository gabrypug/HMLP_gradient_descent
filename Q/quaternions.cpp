#include<iostream>
#include<math.h>
#include"quaternions.h"

using namespace std;

quaternion q_set(){
    quaternion result;
    result.r = 0;
    result.i = 0;
    result.j = 0;
    result.k = 0;
    return result;
}

quaternion q_set(double r, double i, double j, double k) {
    quaternion result;
    result.r = r;
    result.i = i;
    result.j = j;
    result.k = k;
    return result;
}

quaternion q_conj(quaternion q){
    quaternion result;
    result.r = q.r;
    result.i = - q.i;
    result.j = - q.j;
    result.k = - q.k;
    return result;
}

quaternion q_sum(quaternion q1, quaternion q2){
    quaternion result;
    result.r = q1.r + q2.r;
    result.i = q1.i + q2.i;
    result.j = q1.j + q2.j;
    result.k = q1.k + q2.k;
    return result;
}    

quaternion q_diff(quaternion q1, quaternion q2){
    quaternion result;
    result.r = q1.r - q2.r;
    result.i = q1.i - q2.i;
    result.j = q1.j - q2.j;
    result.k = q1.k - q2.k;
    return result;
}

quaternion q_prod(quaternion q1, quaternion q2){
    quaternion result;
    result.r = q1.r*q2.r - q1.i*q2.i - q1.j*q2.j - q1.k*q2.k;
    result.i = q1.j*q2.k - q2.j*q1.k + q1.i*q2.r + q2.i*q1.r;
    result.j = q2.i*q1.k - q1.i*q2.k + q1.j*q2.r + q2.j*q1.r;
    result.k = q1.i*q2.j - q2.i*q1.j + q1.k*q2.r + q2.k*q1.r;
    return result;
}

quaternion q_prod(double r, quaternion q){
    quaternion result;
    result.r = r*q.r;
    result.i = r*q.i;
    result.j = r*q.j;
    result.k = r*q.k;
    return result;
}

quaternion q_sigmoid(quaternion q){
    quaternion result;
    result.r = (1.0/(1.0 + expf(-q.r)));
    result.i = (1.0/(1.0 + expf(-q.i)));
    result.j = (1.0/(1.0 + expf(-q.j)));
    result.k = (1.0/(1.0 + expf(-q.k)));
    return result;
}

quaternion q_diffsig(quaternion q){
    quaternion result;
    result.r = q.r * (1 - q.r);
    result.i = q.i * (1 - q.i);
    result.j = q.j * (1 - q.j);
    result.k = q.k * (1 - q.k);
    return result;
}

std::ostream& operator<<(std::ostream& out, const quaternion& q)
{
    out <<"(" << q.r << " + i" << q.i << " + j" << q.j << " + k" << q.k <<")";
    return out;
}