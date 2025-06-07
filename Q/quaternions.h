#ifndef QUATERNIONS_H
#define QUATERNIONS_H

using namespace std;

typedef struct Tquaternion{
    double r;
    double i;
    double j;
    double k;
} quaternion;

quaternion q_set();

quaternion q_set(double r, double i, double j, double k);

// Quaternionic conjugate
quaternion q_conj(quaternion q);

// Quaternionic sum
quaternion q_sum(quaternion q1, quaternion q2);

// Quaternionic difference
quaternion q_diff(quaternion q1, quaternion q2);

// Quaternionic product
quaternion q_prod(quaternion q1, quaternion q2);

// Quaternionic product
quaternion q_prod(double r, quaternion q);

// Quaternionic sigmoidal
quaternion q_sigmoid(quaternion q);

// Quaternionic derivate
quaternion q_diffsig(quaternion q);

// Print quaternion
std::ostream& operator<<(std::ostream& out, const quaternion& q);

#endif