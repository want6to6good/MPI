#include<iostream>
#include<fstream>
#include<cmath>
#include<stdio.h>
#include<mpi.h>
#include<stdlib.h>
#include<time.h>

using namespace std;
#define PROGRESS_NUM 6
const int n = 1000;
float m[n][n];
int getEnd(int rank)
{
	int t_end;
	if (rank == PROGRESS_NUM - 1) t_end = n - 1;
	else t_end = (rank + 1) * (n / PROGRESS_NUM) - 1;
	return t_end;
}
int main()
{
	std::ifstream inp("input.txt");
	int pn;
	inp >> pn;
	for (int i = 0; i < n; i++)
		for (int j = 0; j < n; j++)
			inp >> m[i][j];
	MPI_Init(0, 0);
	int rank;
	//MPI_Status status;
	double head, tail;
	head = MPI_Wtime();
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	// 块划分
	int t_begin = rank * (n / PROGRESS_NUM);
	int t_end = getEnd(rank);
	if (rank == 0) 
	{
		// 分工
		int tasknum, t1, t2;
		for (int j = 1; j < PROGRESS_NUM; j++)
		{
			t1 = j * (n / PROGRESS_NUM);
			t2 = getEnd(j);
			tasknum = n * (t2 - t1 + 1);
			MPI_Send(&m[t1][0], tasknum, MPI_FLOAT, j, n + 1, MPI_COMM_WORLD);
		}
	}
	else
		MPI_Recv(&m[t_begin][0], n * (t_end - t_begin + 1), MPI_FLOAT, 0, n + 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	MPI_Barrier(MPI_COMM_WORLD);
	float d_temp, e_temp;  // 使用d_temp（除法temp）和e_temp（消去temp）来减少跳跃的内存访问次数
	for (int k = 0; k < n; k++)
	{
		if (rank == 0)
		{	//0号进程负责除法
			d_temp = m[k][k];
			for (int j = k + 1; j < n; j++)
				m[k][j] /= d_temp;
			m[k][k] = 1.0;
			for (int j = 1; j < PROGRESS_NUM; j++)
				MPI_Send(&m[k][0], n, MPI_FLOAT, j, k + 1, MPI_COMM_WORLD);
		}
		else
			MPI_Recv(&m[k][0], n, MPI_FLOAT, 0, k + 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		if (t_end > k)
		{
			for (int i = std::max(k + 1, t_begin); i <= t_end; i++)
			{  // 防溢出，块划分思想
				e_temp = m[i][k];
				for (int j = k + 1; j < n; j++)
					m[i][j] -= e_temp * m[k][j];
				m[i][k] = 0;
				if (i == k + 1 && rank)
					MPI_Send(&m[i][0], n, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
			}
		}
		if (!rank && t_end < k + 1 && n > k + 1)
			MPI_Recv(&m[k + 1][0], n, MPI_FLOAT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	}
	MPI_Barrier(MPI_COMM_WORLD);
	tail = MPI_Wtime();
	MPI_Finalize();
	if (!rank)
		cout << "MPI_GE_Cycle：" << (tail - head) * 1000 << " ms" << std::endl;
	return 0;

}