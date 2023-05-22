#include<iostream>
#include<fstream>
#include<cmath>
#include<stdio.h>
#include<mpi.h>
#include<stdlib.h>
#include<time.h>
#include<algorithm>

using namespace std;
#define PROGRESS_NUM 6

const int n = 1000;
float m[n][n];


int main()
{
	ifstream inp("input.txt");
	int pn;
	inp >> pn;
	for (int i = 0; i < n; i++) 
		for (int j = 0; j < n; j++) 
			inp >> m[i][j];
	MPI_Init(0, 0);
	double head, tail;
	head = MPI_Wtime();
	int rank;
	MPI_Status status;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	if (rank == 0)
	{  // 任务划分
		for (int j = 1; j < PROGRESS_NUM; j++)
		{
			for (int i = j; i < n; i += PROGRESS_NUM - 1)  // 循环划分
				MPI_Send(m[i], n, MPI_FLOAT, j, i, MPI_COMM_WORLD);
		}
	}
	else
	{
		for (int i = rank; i < n; i += PROGRESS_NUM - 1)
			MPI_Recv(m[i], n, MPI_FLOAT, 0, i, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	}
	MPI_Barrier(MPI_COMM_WORLD);
	float d_temp, e_temp;  // 使用d_temp（除法temp）和e_temp（消去temp）来减少跳跃的内存访问次数
	for (int k = 0; k < n; k++)
	{
		if (rank == 0)
		{	//0号进程负责除法部分
			d_temp = m[k][k];
			for (int j = k + 1; j < n; j++)
				m[k][j] /= d_temp;
			m[k][k] = 1.0;
			for (int j = 1; j < PROGRESS_NUM; j++)
				MPI_Send(m[k], n, MPI_FLOAT, j, n + k + 1, MPI_COMM_WORLD);
		}
		else
			MPI_Recv(m[k], n, MPI_FLOAT, 0, n + k + 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		if (rank != 0)
		{
			int index = rank;
			while (index < k + 1)
				index += PROGRESS_NUM - 1;  // 预处理
			for (int i = index; i < n; i += PROGRESS_NUM - 1)
			{
				e_temp = m[i][k];
				for (int j = k + 1; j < n; j++)
					m[i][j] -= e_temp * m[k][j];
				m[i][k] = 0;
				if (i == k + 1)
					MPI_Send(m[i], n, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
			}
		}
		if (!rank && k < n - 1)
			MPI_Recv(m[k + 1], n, MPI_FLOAT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	}
	MPI_Barrier(MPI_COMM_WORLD);
	tail = MPI_Wtime();
	MPI_Finalize();
	if (!rank) 
		cout << "MPI_GE_Cycle：" << (tail - head) * 1000 << " ms" << std::endl;
	return 0;
}