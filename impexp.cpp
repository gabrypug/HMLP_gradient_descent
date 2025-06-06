#include"impexp.h"

using namespace std;


uint32_t countrows(char *filename)
{
    // count the number of lines in the file called filename                                    
    FILE *fp;
    uint64_t ch=0;
    uint64_t lines=0;

    fp = fopen(filename , "r");
    if(fp == NULL) {
       perror("Error opening file");
       return(EXIT_FAILURE);
    }

    while ((ch = fgetc(fp)) != EOF)
    {
        if (ch == '\n')
            lines++;
    }
    fclose(fp);
    return lines;
}

uint32_t countcolumns(char *filename)
{
    // count the number of lines in the file called filename                                    
    FILE *fp;
    uint64_t k = 1;
    uint64_t dim = 10000;
    char str[dim];
    
    /* opening file for reading */
    fp = fopen(filename, "r");
    if(fp == NULL) {
       perror("Error opening file");
       return(EXIT_FAILURE);
    }

    fgets(str, dim, fp);

    for(uint64_t i = 0; i < dim; i++){
        if(str[i] != '\0'){
            if(str[i] == ' ') k++;
        }
        else break;
    }
    fclose(fp);
    return k;
}

double** readMatrixFromFile(char* filename)
{
    FILE *file;
    uint64_t num_rows, num_cols;

    num_rows = countrows(filename);
    // printf("Number of rows: %d\n",num_rows);
    num_cols = countcolumns(filename);
    //printf("Number of columns: %d\n",num_cols);
    //printf("\n");
    
    file = fopen(filename, "r");
    if(file == NULL) {
       perror("Error opening file");
       exit(EXIT_FAILURE);
    }

//matrix
    double** mat=(double**)malloc(num_rows*sizeof(double*));
    for(uint64_t i=0;i<num_rows;++i)
        mat[i]=(double*)malloc(num_cols*sizeof(double));

    for(uint64_t i = 0; i < num_rows; i++)
    {
        for(uint64_t j = 0; j < num_cols; j++)
        {
            fscanf(file, "%lf", &mat[i][j]);
            // printf("%lf ",mat[i][j]);
        }
        // printf("\n");
    }
    fclose(file);
    return mat;
}

void freeMatrix(double** matrix, uint32_t size1)
{
    for(uint64_t i = 0 ; i < size1; i++){
        free(matrix[i]);
    }
    free(matrix);
    matrix = nullptr;

}

