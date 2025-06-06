#ifndef IMPEXP_H
#define IMPEXP_H

#include<iostream>
#include<fcntl.h>
#include <bits/stdc++.h>
#include<time.h>


uint32_t countrows(char *filename);

uint32_t countcolumns(char *filename);

double** readMatrixFromFile(char *filename);

void freeMatrix(double** matrix, uint32_t size1);

#endif